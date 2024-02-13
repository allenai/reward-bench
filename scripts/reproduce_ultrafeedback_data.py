from datasets import load_dataset

ultrafeedback = load_dataset("openbmb/UltraFeedback")
shp = load_dataset("stanfordnlp/SHP")
anthropic_hh = load_dataset("Anthropic/hh-rlhf")
openai_summarize = load_dataset("openai/summarize_from_feedback", "comparisons")