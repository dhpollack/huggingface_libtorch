cd $(dirname $0)/..

mkdir -p models && cd models

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=1i0rr-ogZ2MDYPpUMBsg-2PV7zVddivJ0" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i0rr-ogZ2MDYPpUMBsg-2PV7zVddivJ0" -O sst2_trained.tar.gz && rm -rf /tmp/cookies.txt

tar xzvf sst2_trained.tar.gz

# run the following command
echo 'run: `python ../scripts/trace_albert.py`'
