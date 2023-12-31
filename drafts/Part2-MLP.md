[⏮ README.md](README.md)

# Episode 3: Building makemore Part 2: MLP

A C++ implementation.

![](https://i.ytimg.com/vi/TCH_1BHY58I/hqdefault.jpg)

Each step of the third episode of *Neural Nets: Zero to Hero*:
[Bulding makemore Part 2: MLP](https://youtu.be/TCH_1BHY58I)
is included.

 * [Intro](#intro)
 * [LogitMLP](#logitmlp)
   - [Embedding lookup table](#embedding-lookup-table)
   - [Hidden layer](#hidden-layer)
   - [Output layer](#output-layer)
   - [Plotting loss](#plotting-loss)
   - [Summary of the network](#summary-of-the-network)
 * [Cross-Entropy](#cross-entropy)
 * [Batch training](#batch-training)
 * [Visualizing the character embedding](#visualizing-the-character-embedding)
 * [Sampling](#sampling)

## Intro

The bigram model is somewhat limited as it only looks at one previous letter at a time. If we increased the context length then the
number of possibilities would explode. Currently there are 27 possible next letters for each of 27 possible previous letters
(the 27th letter is the start or end token `.`), producing a 27x27 matrix of `27 * 27 = 729` probabilities.

Each additional letter of context would multiply this by 27:

| Context length |  Possibilities | Size |
|----------------|----------------|------|
| 1 | 27 x 27 | 729
| 2 | 27 x 27 x 27 = 27^3 | 20K
| 3 | 27 ^ 4 | 531K
| 4 | 27 ^ 5 | 14M
| 5 | 27 ^ 6 | 387M
| 6 | 27 ^ 7 | 10B

Clearly it would be inefficient to simply count all the possibilities: it would take 10B parameters to keep only 6 letters of context.
We want to build towards much longer context lengths, so we first need to drastically reduce the size of the model.

## LogitMLP

### Embedding lookup table

### Hidden layer

### Output layer

### Plotting loss

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


### Summary of the network

Lastly we add a hidden layer, which is just another matrix sandwiched in-between the inputs and outputs, and it has tanh for non-linearity. Oh and first we include an embedding layer. And we add bias to the output.

```c++
template <size_t ContextLength, size_t N, size_t E, size_t H, size_t M>
class LogitMLP {
    public:
        LogitMLP()
            : C_(make_node(Eigen::MatrixXd(N, E))),
            W1_(make_node(Eigen::MatrixXd(ContextLength*E, H))),
            B1_(make_node(Eigen::RowVectorXd(H))),
            W2_(make_node(Eigen::MatrixXd(H, M))),
            B2_(make_node(Eigen::RowVectorXd(M)))

        {
        }

        size_t model_params() const {
            return count_params(C_) + count_params(W1_) + count_params(W2_) + count_params(B2_);
        }

        Node operator()(const Node& input) const {
            return normalize_rows(exp(tanh(row_vectorize(input * C_) * W1_ + B1_) * W2_ + B2_));
        }

        void adjust(double learning_rate) {
            C_->adjust(learning_rate);
            W1_->adjust(learning_rate);
            W2_->adjust(learning_rate);
            B2_->adjust(learning_rate);
        }

    private:
        Node C_;
        Node W1_;
        Node B1_;
        Node W2_;
        Node B2_;
};
```

## Cross-Entropy

Classification

Backward pass more efficient : can fuse kernels etc
clustered mathematical expression

numerical stability: constant offset does not change cross-entropy result

## Batch training

## Visualizing the character embedding

## Sampling

Eventually we want to be able to sample from this model.
It is no longer feasible to cache the probability distributions for all possible outputs.
We can write a general class for sampling from any model based on Node.
It continuously generates the probability distribution for the next output and then samples from that.

[std::discrete_distribution](https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution).


```c++
template <typename F>
class ModelSampler {
    private:
        std::mt19937 rng; // Random number generator

    public:
        // Constructor that takes a precalculated probability matrix
        ModelSampler(const F& func)
            //: rng(std::random_device{}())
            : rng(static_mt19937()), func_(func)
        {}

        // Operator to sample given input
        template <typename Input>
        size_t operator()(const Input& input) {
            Node input_node = make_node(input);
            Node output = func_(input_node);
            Eigen::RowVectorXd row = output->data();
            std::vector<double> prob_vector(row.data(), row.data() + row.size());
            std::discrete_distribution<int> dist(prob_vector.begin(), prob_vector.end());
            return dist(rng);
        }

    private:
        const F& func_;
};
```
