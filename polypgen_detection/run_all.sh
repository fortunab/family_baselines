# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# 1. Florence-2
python3 main_detection.py --model-name florence2

# 2. OWLv2
python3 main_detection.py --model-name owlv2

# 3. Grounding DINO
python3 main_detection.py --model-name grounding_dino

# 4. PaliGemma
python3 main_detection.py --model-name paligemma

# Compare all 4 models
python3 compare_detectors.py

