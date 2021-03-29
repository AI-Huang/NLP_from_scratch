# NLP_from_scratch

NLP_from_scratch, PyTorch NLP tutorials series.

## NLP: Names Classification with RNN

![RNN Model](./assets/rnn_model.png)

### Prepare Data's Pipeline

Index language files by categories:

```Python
def findFiles(path): return glob.glob(path)

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
```

The results (on MacOS) of `all_categories` is:

```bash
['Czech', 'German', 'Arabic', 'Japanese', 'Chinese', 'Vietnamese', 'Russian', 'French', 'Irish', 'English', 'Spanish', 'Greek', 'Italian', 'Portuguese', 'Scottish', 'Dutch', 'Korean', 'Polish']
```

This is not alphabetical due to the `glob` package. However, this order is which the model's one-hot vector output will follow.

## NLP: Names Generation with RNN

![RNN Model for Names Generation](./assets/rnn_model_2.png)

This model is supervised and fed by (category, line, line). Notice that the target line is totally the same with the input line but with different data representation and has an extra `EOS`.

```python
def randomTrainingExample():
    """
    Make category, input, and target tensors from a random category, line pair
    """
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor target_line_tensor
```

## NLP: Translation

Seq2Seq Model

### Data and Data Preprocessing

To read the data file we will split the file into lines, and then split lines into pairs. The files are all English → Other Language, so if we want to translate from Other Language → English I added the `reverse` flag to reverse the pairs.

The data file data/eng-fra.txt contains lines of language pairs from English -> French, e.g.:

```
Go.	Va !
Run!	Cours !
Run!	Courez !
Wow!	Ça alors !
Fire!	Au feu !
Help!	À l'aide !
Jump.	Saute.
```

The delimiter of languages is `'\t'`, so we split each line by `'\t'`:

```Python
pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
```

#### Word2Vec, a word indices representing

After above TODO

### Training, data feeding

In the original code in PyTorch tutorial\[3\], in function `trainIters`:

```Python
training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
```

Function `tensorsFromPair` takes two **global** variables `input_lang` and `output_lang`.

```Python
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
```

We changed it to:

```Python
def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
```

We also moved `device` argument to `train` function:

```Python
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device):
    encoder_hidden = encoder.init_hidden().to(device)
    ...
```

so that for every `init_hidden` function in each seq2seq `nn.module` , the `device` argument is not needed anymore.

## References

3. PyTorch, [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
