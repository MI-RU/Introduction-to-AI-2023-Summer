# Syllabus - Introduction to AI (2023 Summer)
## Goal
이 강의는 인공 지능(AI)과 기계 학습(ML)의 기초부터 시작하여 AI 및 ML의 핵심 개념, 심화 주제에 대해 다루게 될 예정입니다.
OT는 클러스터링, 회귀, 강화 학습 및 데이터 전처리의 중요한 역할과 같은 주요 영역을 다루는 AI 및 ML의 개요로 시작합니다.
1주차 강의는 데이터 스케일링, 베이즈 정리 및 손실 함수(Loss function)과 같은 개념을 포괄적으로 살펴보며 데이터 분석을 위한 확률 이론과 및 지표에 대해 탐구합니다.
2주차 강의에서는 신경망에 초점을 맞춰 퍼셉트론에서 컨볼루션 및 순환 네트워크에 이르기까지 작동 원리를 설명합니다.
2주차 강의의 내용을 확장시켜 3주차 강의에서는 '차원의 저주'에 대한 의미와 및 완화 전략(차원 감소 기술 및 효과적인 데이터 시각화 포함)에 대해 논의합니다.
그런 다음 이 시리즈는 Meetup 4의 시계열 분석으로 이동하여 시계열 데이터를 이해하고 예측하기 위해 ARIMA 및 지수 평활과 같은 다양한 모델을 탐색합니다.
Meetup 5에서는 Markov Chains를 소개하고 강화 학습에 대해 자세히 살펴보며 정책 기반 강화 학습 및 해당 애플리케이션에 중점을 둔 최종 세션인 Meetup 6을 위한 길을 닦습니다.
이 포괄적인 강의 시리즈는 참가자들이 기본 개념에서 최첨단 기술에 이르기까지 AI 및 ML에 대한 확실한 이해를 갖추도록 하는 것을 목표로 합니다. AI 및 ML의 복잡성에 대해 자세히 알아보고자 하는 학생, 전문가에게 적합합니다.

This lecture series takes you on an in-depth journey through the fascinating world of Artificial Intelligence (AI) and Machine Learning (ML). Starting from the basics, we explore the core concepts, techniques, and applications of AI and ML, preparing you for a deep dive into advanced topics.
Meetup 0 initiates the series with an overview of AI and ML, covering key areas like clustering, regression, reinforcement learning, and the critical role of data preprocessing.
Meetup 1 delves into the realm of probability theorems and metrics for data analysis, taking a comprehensive look at concepts like data scaling, Bayes Theorem, and loss calculation.
In Meetup 2, we switch gears to focus on Neural Networks, elucidating their working principles, from perceptrons to convolutional and recurrent networks.
This exploration continues in Meetup 3, where we discuss the 'Curse of Dimensionality', its implications, and mitigation strategies, including dimensionality reduction techniques and effective data visualization.
The series then shifts to time-series analysis in Meetup 4, exploring various models like ARIMA and Exponential Smoothing to understand and forecast time-series data.
In Meetup 5, we introduce Markov Chains and delve further into Reinforcement Learning, paving the way for the final session, Meetup 6, that focuses on Policy-Based Reinforcement Learning and its applications.
This comprehensive lecture series aims to equip participants with a solid understanding of AI and ML, from fundamental concepts to cutting-edge techniques. It's perfect for students, professionals, and enthusiasts eager to dive deep into the intricacies of AI and ML.

## Lectures
### meetup #0 -  Introduction to Artificial Intelligence and Machine Learning
Part 1: What is AI? (20 minutes)
- Definition and scope of Artificial Intelligence (AI)
- Historical overview and milestones in AI development
- Different types of AI: Narrow AI vs. General AI
- Applications of AI in various domains

Part 2: Machine Learning Overview (25 minutes)
- Introduction to Machine Learning (ML)
- Supervised, unsupervised, and reinforcement learning paradigms
- Key components of a machine learning system: data, model, and optimization
- Role of machine learning in AI systems

Part 3: Clustering (20 minutes)
- Definition and concept of clustering
- Different clustering algorithms: K-means, hierarchical clustering, DBSCAN
- Use cases and applications of clustering in data analysis

Part 4: Regression (20 minutes)
- Introduction to regression analysis
- Types of regression: linear regression, polynomial regression, and others
- Assessing model performance: evaluation metrics and residual analysis
- Real-world applications of regression analysis

Part 5: Reinforcement Learning (20 minutes)
- Definition and characteristics of reinforcement learning (RL)
- Agent, environment, states, actions, and rewards in RL
- Exploration vs. exploitation trade-off
- Applications of RL in robotics, gaming, and control systems

Part 6: Importance of Preprocessing (15 minutes)
- Why preprocessing is important in data analysis and machine learning
- Data cleaning: handling missing values, outliers, and noise
- Feature scaling and normalization
- Feature engineering and dimensionality reduction

Part 7: Conclusion and Future Directions (20 minutes)
- Recap of key concepts covered in the lecture
- Current trends and advancements in AI and ML
- Ethical considerations and challenges in AI development
- Opportunities and potential future directions in the field


### meetup #1 - Probability Theorems and Metrics for Data Analysis
Part 1: Introduction to Metrics for Data Analysis (15 minutes)
- Importance of metrics in data analysis
- Role of metrics in quantifying data characteristics
- Types of metrics: descriptive, inferential, and predictive
- Applications of metrics in different domains

Part 2: Data Scaling (20 minutes)
- Motivation behind data scaling
- Common data scaling techniques: min-max scaling, standardization, and normalization
- Effects of data scaling on machine learning algorithms
- Considerations and best practices for data scaling

Part 3: Probability Theorems (25 minutes)
- Importance of probability in data analysis
- Concept of likelihood: definition and interpretation
- Probability density function (PDF) and likelihood function
- Maximum likelihood estimation (MLE) and its applications

Part 4: Bayes Theorem (30 minutes)
- Introduction to Bayes Theorem
- Prior probability, likelihood, and posterior probability
- Applications of Bayes Theorem in machine learning and data analysis
- Bayesian inference and its advantages

Part 5: Practice: Calculating Loss (20 minutes)
- Overview of loss functions in data analysis and machine learning
- Common loss functions: mean squared error (MSE), cross-entropy, and others
- Calculation of loss for different tasks: regression, classification, and others
- Impact of loss functions on model training and performance

Part 6: Normalization and the Normal Distribution (25 minutes)
- Introduction to normalization techniques
- Normal distribution and its properties
- Z-score normalization and standardization
- Applications of normal distribution in data analysis and hypothesis testing

Part 7: Probability Theorems and Metrics Practices (25 minutes)
- Calculating Likelihood:
- - Given a dataset and a probability distribution, calculate the likelihood function for the observed data.
Apply the maximum likelihood estimation (MLE) method to estimate the parameters of the distribution.
- Normalization Practice:
- - Select a dataset and apply normalization techniques such as Z-score normalization and min-max scaling.
- - Analyze the impact of normalization on the data distribution and discuss the benefits of normalization.


### meetup #2 - Introduction to Neural Network
Part 1: Perceptron and Artificial Neurons (20 minutes)
- Motivation behind artificial neurons and neural networks
- Perceptron model and its components
- Activation function and thresholding
- Perceptron learning rule and weight updates
- Single-layer perceptron and its limitations

Part 2: Quick Review of Partial Derivatives (15 minutes)
- Importance of partial derivatives in neural networks
- Review of basic concepts of partial derivatives
- Chain rule and its application in gradient calculations

Part 3: Activation Functions (20 minutes)
- Role of activation functions in neural networks
- Step function, sigmoid function, and their properties
- Rectified Linear Unit (ReLU) and its advantages
- Popular activation functions: tanh, softmax, and others

Part 4: Backpropagation Algorithm (30 minutes)
- Introduction to backpropagation
- Forward pass and calculation of output
- Error function and loss calculation
- Backward pass and gradient descent
- Weight updates and learning rate

Part 5: Convolutional Neural Networks (CNN) (30 minutes)
- Motivation behind CNNs and their applications
- Convolutional layers and filter operations
- Pooling layers and downsampling
- Architecture of CNNs: convolutional, pooling, and fully connected layers
- Popular CNN architectures: LeNet, AlexNet, VGG, and ResNet

Part 6: Recurrent Neural Networks (RNN) (25 minutes)
- Introduction to RNNs and their sequential nature
- Recurrent connections and hidden states
- Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)
- Applications of RNNs: sequential data processing, natural language processing, and time series analysis

Part 7: Neural Network Practices (15 minutes)
- Implementing a Multilayer Perceptron (MLP):
- Use a dataset of your choice and implement an MLP using a machine learning library.
- Experiment with different architectures, activation functions, and learning rates.
- Evaluate the performance of the MLP model using appropriate evaluation metrics.


### meetup #3 - Curse of Dimensionality and Dimensionality Reduction
Part 1: Introduction to Curse of Dimensionality (15 minutes)
- Definition and motivation behind the Curse of Dimensionality
- Impact of high-dimensional data on machine learning algorithms
- Challenges in data storage, computational complexity, and model performance

Part 2: Norm in Linear Algebra (20 minutes)
- Introduction to norms and their properties
- Euclidean norm and Manhattan norm
- p-norm and the generalized form of norms
- Application of norms in measuring distance and similarity

Part 3: Projection and Dimensionality Reduction (30 minutes)
- Projection: reducing data from a higher-dimensional space to a lower-dimensional space
- Orthogonal projection and its properties
- Principal Component Analysis (PCA) as a dimensionality reduction technique
- PCA algorithm and its steps: mean centering, covariance matrix, eigendecomposition
- Selecting the number of principal components and explained variance ratio

Part 4: Visualization Techniques (30 minutes)
- Importance of visualization in understanding high-dimensional data
- Scatter plots, pair plots, and parallel coordinate plots for visualizing multi-dimensional data
- t-SNE (t-Distributed Stochastic Neighbor Embedding) for nonlinear dimensionality reduction and visualization
- Advantages and limitations of different visualization techniques

Part 5: Curse of Dimensionality in Machine Learning (20 minutes)
- Impact of high-dimensional data on model complexity and overfitting
- Sparsity of high-dimensional data and the need for more data
- Data preprocessing techniques for addressing the curse of dimensionality
- Feature selection and feature extraction as dimensionality reduction approaches

Part 6: Dimensionality Reduction Practices (25 minutes)
- Implementing PCA:
- Use a dataset of your choice and apply PCA for dimensionality reduction.
- Visualize the explained variance ratio and select the optimal number of principal components.
- Assess the impact of dimensionality reduction on model performance using a supervised learning algorithm.

Part 7: Visualization Practices (20 minutes)
Applying PCA:
Select a high-dimensional dataset and apply PCA to reduce its dimensionality.
Visualize the reduced data using scatter plots or other suitable visualization techniques.
Analyze the patterns and clusters in the reduced space and discuss the insights gained.


### meetup #4 - Introduction to Time Series Analysis
Part 1: Introduction to Time Series Analysis (15 minutes)
- Definition and characteristics of time series data
- Applications of time series analysis in various domains
- Challenges and considerations in analyzing time series data

Part 2: Time Series Components and Patterns (20 minutes)
- Trend, seasonality, and cyclicity in time series data
- Stationarity and its importance in time series analysis
- Autocorrelation and partial autocorrelation functions
- Identifying and understanding different patterns in time series data

Part 3: Time Series Modeling: ARIMA (30 minutes)
- Introduction to Autoregressive Integrated Moving Average (ARIMA) models
- Stationarity testing and differencing for achieving stationarity
- Identification, estimation, and interpretation of AR, MA, and I components
- Model selection and order determination using AIC and BIC
- Forecasting with ARIMA models

Part 4: Time Series Modeling: Seasonal ARIMA (25 minutes)
- Seasonal components in time series data
- Introduction to Seasonal ARIMA (SARIMA) models
- Incorporating seasonal differencing in SARIMA models
- Model selection and parameter estimation for SARIMA
- Forecasting with SARIMA models

Part 5: Time Series Modeling: Exponential Smoothing (20 minutes)
- Introduction to Exponential Smoothing models
- Simple, Holt's, and Winter's exponential smoothing techniques
- Trend and seasonality smoothing factors
- Model selection and forecasting with Exponential Smoothing

Part 6: Time Series Modeling: ARIMA vs. Exponential Smoothing (20 minutes)
- Comparison of ARIMA and Exponential Smoothing models
- Strengths and limitations of each approach
- Choosing the appropriate model for different time series data
- Combining ARIMA and Exponential Smoothing models

Part 7: Time Series Analysis Practices (30 minutes)
- Implementing ARIMA Models:
- - Use a time series dataset of your choice and implement ARIMA modeling using a statistical or machine learning library.
Perform necessary data preprocessing, identify the optimal order of the ARIMA model, and forecast future values.
Evaluate the performance of the model using appropriate evaluation metrics.
- Implementing Exponential Smoothing Models:
- - Select a time series dataset and implement Exponential Smoothing models such as Simple, Holt's, or Winter's smoothing.
Train the models, tune the smoothing factors, and compare the forecasting accuracy of different models.


### meetup #5 - Markov Chains and Reinforcement Learning
Part 1: Markov Chains (30 minutes)
- Introduction to Markov Chains and their properties
- State space, transition probabilities, and Markov property
- Markov Chain representation using state transition matrix
- Stationary distribution and limiting behavior of Markov Chains
- Applications of Markov Chains in various domains

Part 2: Introduction to Reinforcement Learning (20 minutes)
- Definition and key components of reinforcement learning (RL)
- Agent, environment, states, actions, and rewards
- Exploration vs. exploitation trade-off
- Sequential decision-making and the Markov Decision Process (MDP)
- Value function and policy in MDPs

Part 3: Value-Based Reinforcement Learning (30 minutes)
- Introduction to value-based RL algorithms
- Q-learning: estimating the action-value function
- Q-value iteration and the update rule
- Exploration techniques: epsilon-greedy, softmax, and UCB
- Off-policy learning and the use of a target network

Part 4: Deep Q-Networks (DQN) (25 minutes)
- Introduction to Deep Q-Networks (DQN)
- Q-learning with function approximation
- Deep neural network architecture for Q-learning
- Experience replay and its benefits
- Training and convergence of DQN


### meetup #6 - Policy-Based Reinforcement Learning and Applications
Part 1: Introduction to Policy-Based Reinforcement Learning (20 minutes)
- Recap of value-based reinforcement learning
- Limitations of value-based methods
- Introduction to policy-based reinforcement learning
- Advantages and trade-offs of policy-based methods

Part 2: Policy Parameterization and Policy Gradients (30 minutes)
- Policy parameterization and function approximation
- Policy gradient theorem and its derivation
- Policy gradient algorithms: REINFORCE and Proximal Policy Optimization (PPO)
- Advantages of using policy gradients for stochastic policies

Part 3: Actor-Critic Methods (35 minutes)
- Introduction to actor-critic methods
- Combining value-based and policy-based approaches
- Actor-Critic architecture and the advantage function
- Advantage Actor-Critic (A2C) and Asynchronous Advantage Actor-Critic (A3C)
- Deep Deterministic Policy Gradient (DDPG) algorithm

Part 4: Proximal Policy Optimization (PPO) (30 minutes)
- Introduction to Proximal Policy Optimization (PPO)
- Policy iteration and surrogate objective functions
- Trust region methods and policy updates
- PPO algorithm and its variants
- Improvements and extensions of PPO
