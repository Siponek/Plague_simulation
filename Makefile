.PHONY: simone
simone:
	python .\mmn_queue_redacted.py --weibull 2 --d 3 --max-t 20000

.PHONY: szymon
szymon:
	python .\mmn_queue.py --weibull 2 --sample_size 3 --max-t 20000

