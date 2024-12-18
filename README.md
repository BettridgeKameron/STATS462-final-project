# STAT 462 project
The presentation and report are the PDFS in the previous directory, and also should be uploaded seperately.

./copora/ has the training data that was used
./models/ has trained models
`_enc` appeneded to name is an encrypted version of the non appended file using a random subsitution. 

# Usage
## Train on a file
./mc-cryptanalysis.py train training.txt model.json

## Train on direct text
./mc-cryptanalysis.py train "your training text here" model.json

## Encrypt a file
./mc-cryptanalysis.py encrypt message.txt output.txt

## Encrypt direct text
./mc-cryptanalysis.py encrypt "your message here" output.txt

## Decrypt a file
./mc-cryptanalysis.py decrypt encrypted.txt model.json

## Decrypt direct text
./mc-cryptanalysis.py decrypt "xvkkf nfikw" model.json
