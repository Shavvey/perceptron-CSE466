MAIN=src/main.py
DATA=src/data.py

run: $(MAIN)
	python3 $(MAIN)

data: $(DATA)
	python3 $(DATA)
