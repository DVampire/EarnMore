# PortfolioManagement
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/microsoft/qlib.git && cd qlib
python setup.py install
cd ..
pip install -r requirements.txt
```

# RUN
```
1. make scripts
sh tools/pipeline_mask_sac_dj30_example.sh

2. make pipeline
python tools/pipeline.py

3. run pipeline
sh tools/pipeline.sh
```

# References

ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL

RL-Adventure: https://github.com/higgsfield/RL-Adventure

Qlib: https://github.com/microsoft/qlib