We want to try out the loss reporting functionality as discussed in [gensim issue #999](https://github.com/RaRe-Technologies/gensim/issues/999) and (partially) implemented in [gensim PR #1201](https://github.com/RaRe-Technologies/gensim/pull/1201), using the `develop` branch of gensim as of 2017-07-23 (commit [da383bf4a4046b134d95d9085eedb163dd5e0c46](https://github.com/RaRe-Technologies/gensim/commit/da383bf4a4046b134d95d9085eedb163dd5e0c46)), and the first 100k posts (> 3M words) from the [One Million Posts Corpus](https://ofai.github.io/million-post-corpus) as data.

The [test script is here](test_doc2vec_loss.py)

```bash
MAXALPHA=0.1
python test_doc2vec_loss.py --maxalpha "$MAXALPHA" 2>&1 | tee log_maxalpha_"$MAXALPHA".txt
mv word2vec_loss.png word2vec_loss_maxalpha_"$MAXALPHA".png
```

The [log file is here](log_maxalpha_0.1.txt).

```bash
$ grep '^After epoch' log_maxalpha_"$MAXALPHA".txt
After epoch 1: latest training loss is 10294905.000000
After epoch 2: latest training loss is 1203828.125000
After epoch 3: latest training loss is 245310.640625
After epoch 4: latest training loss is 166148.406250
After epoch 5: latest training loss is 136867.781250
After epoch 6: latest training loss is 121953.609375
After epoch 7: latest training loss is 111192.812500
After epoch 8: latest training loss is 109865.835938
After epoch 9: latest training loss is 107571.671875
After epoch 10: latest training loss is 107939.546875
```

![] (word2vec_loss_maxalpha_0.1.png)

This is nice, it looks like a typical epoch vs. loss plot: monotonically decreasing, large changes at first and then less and less changes.

With a smaller `max_alpha` however, it looks problematic:

```bash
MAXALPHA=0.05
python test_doc2vec_loss.py --maxalpha "$MAXALPHA" 2>&1 | tee log_maxalpha_"$MAXALPHA".txt
mv word2vec_loss.png word2vec_loss_maxalpha_"$MAXALPHA".png
```

The [log file is here](log_maxalpha_0.05.txt).

```bash
$ grep '^After epoch' log_maxalpha_"$MAXALPHA".txt
After epoch 1: latest training loss is 8382980.000000
After epoch 2: latest training loss is 9314768.000000
After epoch 3: latest training loss is 9695628.000000
After epoch 4: latest training loss is 9800970.000000
After epoch 5: latest training loss is 9851732.000000
After epoch 6: latest training loss is 10716296.000000
After epoch 7: latest training loss is 10940231.000000
After epoch 8: latest training loss is 10656052.000000
After epoch 9: latest training loss is 10707312.000000
After epoch 10: latest training loss is 10796318.000000
```

![] (word2vec_loss_maxalpha_0.05.png)

With these smaller learning rates, the loss ***increases*** over time!?
