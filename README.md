# Recommendation-Of-Movies-Using-Movie-Embeddings
Instead of embedding words, we will embed movies. In particular, if we can embed movies, then similar movies will be close to each other and can be recommended. This line of reasoning is analogous to the distributional hypothesis of word meanings. For words, this roughly translates to words that appear in similar sentences should have similar vector representations. For movies, vectors for two movies should be similar if they are watched by similar people.

Let the total number of movies be M. Let Xi,j be the number of users that liked both movies i and j. We want to obtain vectors v1,...,vi,...,vj,...,vM for all movies such that we minimize the cost $c(v_1,...,v_M) = \sum_{i=1}^{M}\sum_{j=1}^{M}\mathbf{1}{[i\neq j]}(v_i^Tv_j - X{i,j})^2$. Here 1[i≠j] is a function that is 0 when i=j and 1 otherwise.

  - Compute data Xi,j from the movielens (small) dataset and description. Briefly describe your data prep workflow (you can use pandas if needed).

  - Optimize function c(v1,...,vM) over v1,...,vM using gradient descent (using pytorch or tensorflow). Plot the loss as a function of iteration for various choices (learning rates, choice of optimizers etc).

  - Recommend top 10 movies (not vectors or indices but movie names) given movies (a) Apollo 13, (b) Toy Story, and (c) Home Alone . Describe your recommendation strategy. Do the recommendations change when you change learning rates or optimizers? Why or why not?
