
#### How to Run
python main.py --data data/dsd.csv          # saves splits, runs Tier 1
python -m src.tier2_finetune                # fine-tunes LLM (~1-2 hrs on GPU)
python main.py --data data/dsd.csv          # full pipeline with fine-tuned Tier 2