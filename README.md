# Recommendation-Of-Movies-Using-Movie-Embeddings
Instead of embedding words, we will embed movies. In particular, if we can embed movies, then similar movies will be close to each other and can be recommended. This line of reasoning is analogous to the [distributional hypothesis of word meanings](https://en.wikipedia.org/wiki/Distributional_semantics). For words, this roughly translates to words that appear in similar sentences should have similar vector representations. For movies, vectors for two movies should be similar if they are watched by similar people.

Let the total number of movies be M. Let X<sub>i,j</sub> be the number of users that liked both movies i and j. We want to obtain vectors v<sub>1</sub>,...,v<sub>i</sub>,...,v<sub>j</sub>,...,v<sub>M</sub> for all movies such that we minimize the cost $c(v_1,...,v_M) = \sum_{i=1}^{M}\sum_{j=1}^{M}\mathbf{1}{[i\neq j]}(v_i^Tv_j - X{i,j})^2$. Here 1[i≠j] is a function that is 0 when i=j and 1 otherwise.

  - Compute data X<sub>i,j</sub> from the movielens (small) dataset and [description](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). Briefly describe your data prep workflow (you can use '''pandas''' if needed).

  - Optimize function c(v<sub>1</sub>,...,v<sub>M</sub>) over v<sub>1</sub>,...,v<sub>M</sub> using gradient descent (using pytorch or tensorflow). Plot the loss as a function of iteration for various choices (learning rates, choice of optimizers etc).

  - Recommend top 10 movies (not vectors or indices but movie names) given movies (a) Apollo 13, (b) Toy Story, and (c) Home Alone . Describe your recommendation strategy. Do the recommendations change when you change learning rates or optimizers? Why or why not?
