## TSNE USAGE TO COMPARE THE IMPACT OF IMAGE DESCRIPTION ON THE SKIN LESION RECOGNITION
We've used the dataset PAD-UFES-20 to explore the impact of image description on the skin lesion recognition.

# Create enviroment:
```bash
    conda create -n eda-sentences
```
# To activate this environment, use
```bash
    conda activate eda-sentences
```
# Install libraries
To install the necessary libraries just write:

```bash
    pip3 install -r requirements.txt
```

# Run TSNE script
Add your csv file on the data folder and then run the script below:

```bash
    python3 src/scripts/tsne_of_sentences.py
```

# Join diffetent images to compare different sentence's combinations
Just select the wanted images that you want to join.

# To deactivate an active environment, use
```bash
    conda deactivate
```