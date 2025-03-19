from mbrs.metrics import MetricChrF
from mbrs.decoders import DecoderMBR

# Check speedups for BLEU, implementation seems slow
if __name__ == "__main__":
    import time
    import random

    common_english_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "is", "are", "was", "were", "been", "has", "had", "did", "does", "shall",
        "should", "may", "might", "must", "can", "could", "would", "will", "don’t", "won’t",
        "can’t", "couldn’t", "wouldn’t", "shouldn’t", "mightn’t", "mustn’t", "should’ve", "would’ve", "can’t", "hasn’t",
        "haven’t", "hadn’t", "doesn’t", "didn’t", "aren’t", "weren’t", "isn’t", "wasn’t", "isn’t", "wasn’t",
        "because", "before", "after", "since", "until", "while", "whereas", "although", "though", "because",
        "if", "once", "unless", "yet", "or", "nor", "for", "and", "but", "so",
        "either", "neither", "whether", "both", "each", "few", "many", "several", "some", "none",
        "number", "great", "little", "old", "young", "big", "small", "high", "low", "long",
        "short", "early", "late", "far", "near", "fast", "slow", "good", "bad", "right",
        "wrong", "first", "last", "next", "previous", "important", "necessary", "possible", "real",
        "true", "whole", "certain", "sure", "clear", "full", "easy", "hard", "difficult", "simple",
        "complex", "beautiful", "ugly", "cool", "warm", "hot", "cold", "strong", "weak", "safe",
        "dangerous", "open", "closed", "huge", "tiny", "heavy", "light", "new", "old", "young",
        "big", "small", "high", "low", "long", "short", "early", "late", "far", "near",
        "fast", "slow", "good", "bad", "right", "wrong", "first", "last", "next", "previous",
        "important", "necessary", "possible", "real", "true", "whole", "certain", "sure", "clear", "full",
        "easy", "hard", "difficult", "simple", "complex", "beautiful", "ugly", "cool", "warm", "hot",
        "cold", "strong", "weak", "safe", "dangerous", "open", "closed", "huge", "tiny", "heavy"
    ]


    def make_random_sent() -> str:
        n_words = random.randint(1,20)
        sent = random.choices(common_english_words, k = n_words)
        return ' '.join(sent)

    def make_random_sents(N: int) -> list[str]:
        return [make_random_sent() for _ in range(N)]
    
    H = 16
    N = 50
    
    sentences = [make_random_sent() for _ in range(N)]
    HYPOTHESES = [make_random_sents(H) for _ in range(N)]

    
    for workers in [1,2,4,8,16,32]:
        metric = MetricChrF(MetricChrF.Config(word_order = 2, num_workers = workers))
        decoder_cfg = DecoderMBR.Config()
        decoder = DecoderMBR(decoder_cfg, metric)

        print()
        
        for include_source in [True, False]:
            start = time.time()
            for i in range(N):
                mbr_decoded_output = decoder.decode(HYPOTHESES[i], HYPOTHESES[i], source=sentences[i] if include_source else None, nbest=1)
            end = time.time()

            print(f"WORKERS = {workers}, SOURCE INCLUDED: {include_source}, RUNTIME = {end-start:.3f}s")
