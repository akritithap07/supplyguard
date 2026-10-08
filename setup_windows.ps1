python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Setup complete. Put DataCoSupplyChainDataset.csv in data/raw/ then run: python -m src.train"
