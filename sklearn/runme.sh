
d1="ST"
d2="WW"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="ST"
d2="AP"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="ST"
d2="ST"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="WW"
d2="ST"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="WW"
d2="AP"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="WW"
d2="WW"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="AP"
d2="ST"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="AP"
d2="AP"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"

d1="AP"
d2="WW"
echo "$d1 -> $d2"
poetry run python3 sklearn/runme.py "data/out/${d1}.csv" "data/out/${d2}.csv"
