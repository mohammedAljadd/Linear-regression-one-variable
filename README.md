# Linear-regression-one-variable

In this project we will implement linear regression with one variable to fit some randomly generated data to a straight line.

In this project we will use two approaches to solve this problem:

    1) Gradient Descent.
    2) Normal Equations.

We will implement this project with python. In another project we will use scikitlearn library, so this project is to explaine things from scratch.

# Importing libraries and initialization of variables

The number of variables is m = 100, the data are generated randomly. Notice that we have on feature with 100 observations. Do not confuse number of features and number of observations per feature.

    import numpy as np
    import matplotlib.pyplot as plt

    m = 100
    x = np.linspace(0, 10, m).reshape((m, 1)) 
    y = np.matrix(x + np.random.randn(m, 1))
    x = np.matrix(np.hstack((np.ones((m,1)),x)))
    beta = np.random.rand(2,1)
    itterations = 1000;
    alpha = 0.03;
    J_history = np.zeros((itterations,1))
  
J_history will store the cost function value at each itterations.
By convention the first vector of x is all ones.

![alt text](https://online.stat.psu.edu/stat462/sites/onlinecourses.science.psu.edu.stat462/files/05mlr/eq_matrix_notation/index.gif)

Here we will not use additive noise ε.

# Defining functions for computations

Gradient descent is shown in the next image :

![alt text](https://miro.medium.com/max/450/1*8Omixzi4P2mnqdsPwIR1GQ.png)

    def cost(beta):
        h = x*beta
        error = h-y
        J = 1/(2*m)*np.dot(error.transpose(),error)
        return  J


    def gradient(x,y,beta):
        grad = (1/m) * (x.transpose()) * (x*beta-y)
        return grad

    def gradientDescent(x,y,alpha,beta,itterations):
        for i in range(itterations):
            beta = beta - alpha*gradient(x,y,beta)
            J_history[i] = cost(beta) 
        return beta
        
  # Solution
    optimal_beta,J_history = gradientDescent(x,y,alpha,beta,itterations)
    minimal_cost = cost(beta)
    print('\u03B20=', optimal_beta[0], ',\u03B21=',optimal_beta[1],'\nJ(\u03B20,\u03B21)=',minimal_cost)
    
β0= [[0.11963282]] ,β1= [[0.97293654]] <br/>
J(β0,β1)= [[0.44146884]]
    
# Plot of the cost function according to number of itterations

    plt.plot(J_history)
    plt.xlabel('number of itterations')
    plt.ylabel('Cost function J')
    plt.show()
    
![alt text](https://github.com/mohammedAljadd/Linear-regression-one-variable/tree/main/plots/j_history.PNG)

# Plot of fitting line and the data


    plt.plot(x[:,1],x*optimal_beta,label='fitting line')
    plt.xlabel('x inputs')
    plt.ylabel('y outputs')
    plt.title('Fitting line and data')
    plt.plot(x[:,1], y, 'o',label='training data')
    plt.legend()
    
![alt text](https://github.com/mohammedAljadd/Linear-regression-one-variable/tree/main/plots/fit.PNG)

 # Performance of regression 
 
 ![alt text](https://ashutoshtripathicom.files.wordpress.com/2019/01/rsquarecanva2.png)

 
 This factor should be close to 1.
 
    y_variance = len(y)*np.var(y)
    sum_squared_errors = (2*m)*cost(optimal_beta)
    Performance = 1 - ( sum_squared_errors )/(y_variance)
    print('The performance R is ',Performance) 
    
 The performance R is  [[0.91348857]]

# 3D plot of the cost function 

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    n = 200
    Xs, Ys = np.meshgrid(np.linspace(-30, 30, n), np.linspace(-30, 30, n))
    Zs = np.array([cost(np.matrix( (t0, t1) ).T) for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
    Zs = np.reshape(Zs, Xs.shape)

    fig = plt.figure(figsize=(7,7))
    ax = fig.gca(projection="3d")
    ax.set_xlabel('\u03B20')
    ax.set_ylabel('\u03B21')
    ax.set_zlabel('J(\u03B2) ')
    ax.view_init(elev=35, azim=40)
    ax.plot_surface(Xs, Ys, Zs, cmap=cm.jet,alpha=1) # colormap : jet , alpha : degree of transparency (0-1)
    
![alt text](https://github.com/mohammedAljadd/Linear-regression-one-variable/tree/main/plots/3d_cost.PNG)

# Contour plot of the cost function

    ax = plt.figure().gca()
    sol = ax.plot(optimal_beta[0], optimal_beta[1], 'r*',label="Optimal solution")
    ax.set_xlabel('\u03B20')
    ax.set_ylabel('\u03B21')
    cntr = plt.contour(Xs, Ys, Zs, np.logspace(-10, 10, 50))
    h1,_ = cntr.legend_elements()
    ax.legend([h1[0]], ['Contours'])
    
![alt text](https://github.com/mohammedAljadd/Linear-regression-one-variable/tree/main/plots/countour.PNG)

The red point is the minimum of our cost function.

# Predictition of a value

    input = 20
    print('input value :',input,'\nPredicted output :',np.matrix((1,input))*optimal_beta)
    
input value : 20 <br/>
Predicted output : [[19.57836372]]

# Normal equation using QR decomposition

Any matrix can be represented as a product of an orthogonal matrix Q and an upper triangular matrix R. In our case the b vector is beta.

![alt text](https://miro.medium.com/max/340/1*A9N6Y-qrSSJ8KDQNIUpOjQ.png)

    X = x
    Q, R = np.linalg.qr(X)
    invR = np.linalg.inv(R)
    Qt = Q.T
    thetaNormal = invR*Qt*y
    print('Gradient descent:',thetaNormal[0],thetaNormal[1],'\nNormal euqation: ',optimal_beta[0],optimal_beta[1])

If we compare beta values using gradient descent and the normal equation we get :

Gradient descent: [[0.11918074]] [[0.97300452]] <br/>
Normal euqation:  [[0.11963282]] [[0.97293654]]

