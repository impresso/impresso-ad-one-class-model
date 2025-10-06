python train_if.py \ 
  --ads compiled_corpus_ads_articles_10000_filtered.jsonl \
  --out_dir artifacts \  
  --model Alibaba-NLP/gte-multilingual-base \
  --pca_dim 128 \                        
  --tail_q 0.10 \        
  --per_lang  


python filter_if.py \
  --mix mixed.jsonl \
  --artifacts artifacts \
  --out_nonads results/nonads.jsonl \
  --out_ads results/ads_remaining.jsonl \
  --uncertain_band 0.01 \
  --out_uncertain results/uncertain.jsonl


# model approach:
pip install -U transformers torch

python xgenre_promo_test.py \
  --in mixed.jsonl \
  --out xgenre_scored.jsonl \
  --ad_threshold 0.60 \
  --batch_size 16 \
  --chunk_words 180 --pool max \
  --min_words 2