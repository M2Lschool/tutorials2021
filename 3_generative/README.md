# Colab for the Generative Models tutorial at M2L 2021

**Authors**: [Fabio Viola](https://scholar.google.co.uk/citations?user=-cCry1cAAAAJ&hl=en) and [Marco Ciccone](https://marcociccone.github.io/)

**Description**:
In this tutorial you will learn how to implement Variational AutoEncoders. We will focus on Conditional VAEs that are generative models that allow one to model the data distribution and generate specific samples conditioned on a given context.

Additionally, we will also briefly explore some of the best practices for efficiently scaling up the computation over multiple devices.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m2lschool/tutorials2021/blob/main/3_generative/VAE_Tutorial_Start.ipynb)

The tutorial is structured in two main parts:
- Part I: *Warmup*
    - Enabling TPUs in colab
    - Handling nested data structures using tree utilities in JAX
    - Distributing computation over multiple devices using `pmap`

- Part II: *Amortized variational inference (VAEs)*
    - Training VAEs optimizing ELBO
    - Training  $\beta$-VAEs using KL annealing
    - Training VAEs using constraint optimization (GECO)

All exercises will be implemented using [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku).

**Credits**
This tutorial is adapted from the [EEML 2019](https://github.com/eemlcommunity/PracticalSessions2019/tree/master/unsupervised) and [EEML 2020](https://github.com/eemlcommunity/PracticalSessions2020/tree/master/unsup) Unsupervised Learning tutorials authored by Mihaela Rosca, David Szepesvari and Stanislaw Jastrzebski.

Designed for education purposes. Please do not distribute without permission. Write at organizers@m2lschool.org if you have any question.

You are welcome to reuse this material in other courses or schools, but please reach out to organizers@m2lschool.org if you plan to do so. We would appreciate it if you could acknowledge that the materials come from M2L 2020 and give credits to the authors. Also please keep a link in your materials to the original repo, in case updates occur.

MIT License

Copyright (c) 2020 m2lschool

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
