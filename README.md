# makemore-cpp-2023

A C++ implementation of
[karpathy/makemore](https://github.com/karpathy/makemore).
Each step of the second episode of *Neural Nets: Zero to Hero*:
[The spelled-out intro to language modeling: building makemore](
https://youtu.be/PaCmpygFfXo)
is included.

 * Bigram Language Model
   - [matplotlib-cpp](#matplotlib-cpp)
   - [Bigram Frequencies](#bigram-frequencies)
   - [Multinomial Sampler](#multinomial-sampler)
   - [Broadcasting Rules](#broadcasting-rules)

   - [OneHot Encoding](#onehot-encoding)
   - [Loss Plot](#loss-plot)

## Bigram Language Model

### matplotlib-cpp

[Cryoris/matplotlib-cpp](https://github.com/Cryoris/matplotlib-cpp)

```bash
$ sudo apt update
$ sudo apt install python3 python3-dev python3-matplotlib
```

### Bigram Frequencies

![Frequency plot](examples/bigram.png)

### Multinomial Sampler

### Broadcasting Rules

[Reductions, visitors and Broadcasting](https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html)

### OneHot Encoding

```c++
static inline Eigen::VectorXd encode_onehot(char c) {
    return OneHot(27, c_to_i(c));
}

static inline Eigen::MatrixXd encode_onehot(const std::string& word) {
    Eigen::MatrixXd matrix(word.size(), 27);

    for (size_t i = 0; i < word.size(); ++i) {
        matrix.row(i) = encode_onehot(tolower(word[i]));
    }

    return matrix;
}

```

We can encode the input string `".emma"` (including start token `'.'`)
and visualize this to make it a little more clear:

```c++
    std::string xs = ".emma";

    auto xenc = encode_onehot(xs);

    plt::imshow(xenc);
```

![OneHot Emma](examples/onehot-emma.png)

### Loss Plot

We can plot the loss values over iterations using [loss_plot.gp](loss_plot.gp):

```gnuplot
set logscale y
set xlabel "Iterations"
set ylabel "Loss"
set terminal svg
set output "loss.svg"
set object 1 rect from screen 0,0 to screen 1,1 behind fillcolor rgb "white" fillstyle solid 1.0
plot "loss.tsv" using 1:2 with lines title "Loss vs Iteration"
```

```bash
$ gnuplot loss_plot.gp
```

![loss.svg](examples/loss.svg)

This shows the loss reduction during training.

