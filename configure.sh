
apt update -y
apt install -y python3 git vim python3.12-venv python3-pip

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

tar -xvf voices.tar.gz

cd /tmp
wget wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xvf piper_amd64.tar.gz
sudo mv piper/piper /usr/bin/piper
sudo cp piper/*.so* /usr/local/lib/
sudo ldconfig
sudo cp -r piper/espeak-ng-data /usr/share/