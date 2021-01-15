**The RTLR learning on a simple recurrent network 
with a piecewise linear activation function**


<img src="https://render.githubusercontent.com/render/math?math=\frac{dh}{dt} = -h %2B W \sigma(h) %2B b">


sigma(x) function:


if x <= -1/lambda:  sigma(x) =  kx - (1 - k / lambda)

if x in (-1/lambda, 1/ lambda): sigma(x) = lambda x
 
if x >= 1/lambda : sigma(x) =  kx + (1 - k / lambda)

![Activation function](img//activation_function.png)

**Examples**

Before
![Network with 4 neurons before the training](img//before.png)

After
![Network with 4 neurons after the training](img//after.png)


Stats
![Network with 4 neurons statistics](img//stats.png)

Before
![Network with 5 neurons before the training](img//before_2.png)

After
![Network with 5 neurons after the training](img//after_2.png)

Stats
![Network with 5 neurons statistics](img//stats_2.png)

Before
![Network with 7 neurons before the training](img//before_3.png)

After
![Network with 7 neurons after the training](img//after_3.png)

Stats
![Network with 7 neurons statistics](img//stats_3.png)

**Comments**

The learning is really slow and can only learn if
the system is already in the ballpark of the optimal parameters.






  
      
      
      
      