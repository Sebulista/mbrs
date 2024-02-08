#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType, Namespace
from typing import Iterable

import torch
from tabulate import tabulate, tabulate_formats
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    M2M100ForConditionalGeneration,
)

from mbrs import timer


def buffer_lines(input_stream: Iterable[str], buffer_size: int = 64):
    buf: list[str] = []
    for i, line in enumerate(tqdm(input_stream)):
        buf.append(line.strip())
        if (i + 1) % buffer_size == 0:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # fmt: off
    parser.add_argument("input", nargs="?", default="-",
                        type=FileType("r", encoding="utf-8"),
                        help="Input file. If not specified, read from stdin.")
    parser.add_argument("--output", "-o", default="-", type=FileType("w"),
                        help="Output file.")
    parser.add_argument("--model", "-m", type=str, default="facebook/m2m100_418M",
                        help="Model name or path.")
    parser.add_argument("--num-candidates", "-n", type=int, default=1,
                        help="Number of candidates to be returned.")
    parser.add_argument("--sampling", "-s", type=str, default="",
                        choices=["eps"],
                        help="Sampling method.")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size.")
    parser.add_argument("--epsilon", "-e", type=float, default=0.02,
                        help="Cutoff parameter for epsilon sampling.")
    parser.add_argument("--lang-pair", "-l", type=str, default="en-de",
                        help="Language name pair. Some models like M2M100 uses this information.")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum length of an output sentence.")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force to use CPU.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="No report statistics.")
    parser.add_argument("--report", type=str, default="rounded_outline",
                        choices=tabulate_formats,
                        help="Report runtime statistics.")
    parser.add_argument("--width", "-w", type=int, default=1,
                        help="Number of digits for values of float point.")
    # fmt: on
    return parser.parse_args()


def main(args: Namespace) -> None:

    src_lang, tgt_lang = tuple(args.lang_pair.split("-"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if torch.cuda.is_available() and not args.cpu:
        model.cuda()
        if args.fp16:
            model.half()
        elif args.bf16:
            model.bfloat16()

    additional_input_attrs = {}
    if isinstance(model, M2M100ForConditionalGeneration):
        tokenizer.src_lang = src_lang
        additional_input_attrs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_lang)

    generation_kwargs = {}
    if args.sampling == "eps":
        generation_kwargs["do_sample"] = True
        generation_kwargs["epsilon_cutoff"] = args.epsilon
    else:
        generation_kwargs["num_beams"] = max(args.beam_size, args.num_candidates)

    generation_config = GenerationConfig(
        max_length=args.max_length,
        num_return_sequences=args.num_candidates,
        **generation_kwargs,
    )

    def generate(inputs: list[str]) -> list[str]:
        model_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(
            device=model.device
        )
        model_inputs |= additional_input_attrs
        with timer.measure("generate"):
            model_outputs = model.generate(
                **model_inputs, generation_config=generation_config
            )
        return tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

    num_sentences = 0
    for lines in buffer_lines(args.input, buffer_size=args.batch_size):
        for output in generate(lines):
            print(output.strip(), file=args.output)
            num_sentences += 1

    if not args.quiet:
        statistics = timer.aggregate().result(num_sentences)
        table = tabulate(
            statistics, headers="keys", tablefmt=args.report, floatfmt=f".{args.width}f"
        )
        print(table)


def cli_main():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
