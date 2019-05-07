Tạo neuron network cơ bản

++ Dữ liệu
- Dữ liệu X đầu vào là cặp 2 giá trị (tọa độ x và tọa độ y)
- Dữ liệu đầu ra là nhãn cho cặp dữ liệu là 0 hoặc 1 (red or blue)

++ Model
- dữ liệu đầu vào có 2 giá trị -> input layer có 2 nodes
- dữ liệu đầu ra có 2 giá trị là 0 hoặc 1 -> có thể chọn ouput layer là 1 node có giá trị 0,1 nhưng nên chọn 2 node để dễ dàng mở rộng sau này.

![alt model layer](http://www.wildml.com/wp-content/uploads/2015/09/nn-from-scratch-3-layer-network-1024x693.png)

- Chọn số chiều (số node) của hidden layer. Càng nhiều chiều, phương trình càng phức tạp và càng phù hợp với dữ liệu.  Nhưng càng nhiều chiều, cần tính toán càng nhiều hơn để đưa ra được dự đoán, cần nhiều network parameter, dễ bị tràn dữ liệu.

- Chọn activation function (là hàm phi tuyến xử lý input để cho ra output thuộc khoảng giá trị (0,1)) cho hidden layer. các loại activation function thường dùng là (tanh, sigmoid, ReLUs). Ở đây dùng sigmoid function.

- Đồ thị các loại activation function:

![alt activation-function](https://theclevermachine.files.wordpress.com/2014/09/nnet-error-functions2.png?w=700&h=352)

- Sử dụng thêm hàm softmax để xử lý giá trị ra theo kiểu xác suất

++ Phương pháp:
- Phương pháp sử dụng ở đây là forward propagation (lan truyền tiến)

![alt forward propagation](http://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D++z_1+%26+%3D+xW_1+%2B+b_1+%5C%5C++a_1+%26+%3D+%5Ctanh%28z_1%29+%5C%5C++z_2+%26+%3D+a_1W_2+%2B+b_2+%5C%5C++a_2+%26+%3D+%5Chat%7By%7D+%3D+%5Cmathrm%7Bsoftmax%7D%28z_2%29++%5Cend%7Baligned%7D&bg=ffffff&fg=000&s=0)

- zi là output của layer i, ai là output sau khi được xử lý với activation function
- wi (trọng số), bi (độ lệch bias) là các tham số của mạng trong quá trình train

- giả sử hidden layer có 500 nodes, x1 hiện tại là ma trận 1x2 (vector 2 chiều) thì w1 (trọng số ở đầu vào hidden layer) là ma trận 2x500 và b1 phải là ma trận 1x500 (vector 500 chiều) và w2 (trọng số ở đầu vào output layer) là ma trận 500x2 và b2 phải là ma trận 1x2 (vector 2 chiều)

- Nhiệm vụ là phải tìm ra được w1, b1, w2, b2 để giảm thiểu sai số (giảm cost function/loss function). Cost function được sử dụng trong logistic regression và NN là cross-entropy

![alt cross-entropy](http://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D++L%28y%2C%5Chat%7By%7D%29+%3D+-+%5Cfrac%7B1%7D%7BN%7D+%5Csum_%7Bn+%5Cin+N%7D+%5Csum_%7Bi+%5Cin+C%7D+y_%7Bn%2Ci%7D+%5Clog%5Chat%7By%7D_%7Bn%2Ci%7D++%5Cend%7Baligned%7D++&bg=ffffff&fg=000&s=0)

- Phương pháp để tối ưu cost function là gradient descent.

- Sigmoid function:

![alt sigmoid-function](https://s0.wp.com/latex.php?latex=%5CLarge%7B%5Cbegin%7Barray%7D%7Brcl%7D+g_%7B%5Ctext%7Blogistic%7D%7D%28z%29+%3D+%5Cfrac%7B1%7D%7B1+%2B+e%5E%7B-z%7D%7D%5Cend%7Barray%7D%7D&bg=ffffff&fg=4e4e4e&s=0)

- Đạo hàm của sigmoid function:

![alt graidient of sigmoid function](https://s0.wp.com/latex.php?latex=%5CLarge%7B%5Cbegin%7Barray%7D%7Brcl%7D+g%27_%7B%5Ctext%7Blogistic%7D%7D%28z%29+%26%3D%26+%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial+z%7D+%5Cleft+%28+%5Cfrac%7B1%7D%7B1+%2B+e%5E%7B-z%7D%7D%5Cright+%29+%5C%5C++%26%3D%26+%5Cfrac%7Be%5E%7B-z%7D%7D%7B%281+%2B+e%5E%7B-z%7D%29%5E2%7D+%5Ctext%7B%28chain+rule%29%7D+%5C%5C++%26%3D%26+%5Cfrac%7B1+%2B+e%5E%7B-z%7D+-+1%7D%7B%281+%2B+e%5E%7B-z%7D%29%5E2%7D+%5C%5C++%26%3D%26+%5Cfrac%7B1+%2B+e%5E%7B-z%7D%7D%7B%281+%2B+e%5E%7B-z%7D%29%5E2%7D+-+%5Cleft+%28+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D+%5Cright+%29%5E2+%5C%5C++%26%3D%26+%5Cfrac%7B1%7D%7B%281+%2B+e%5E%7B-z%7D%29%7D+-+%5Cleft+%28+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D+%5Cright+%29%5E2+%5C%5C++%26%3D%26+g_%7B%5Ctext%7Blogistic%7D%7D%28z%29-+g_%7B%5Ctext%7Blogistic%7D%7D%28z%29%5E2+%5C%5C++%26%3D%26+g_%7B%5Ctext%7Blogistic%7D%7D%28z%29%281+-+g_%7B%5Ctext%7Blogistic%7D%7D%28z%29%29+%5Cend%7Barray%7D%7D&bg=ffffff&fg=4e4e4e&s=0)

- Để xử lý đạo hàm trong NN ta dùng thuật toán backpropagation:
    + Tính các đầu ra từ đầu đến cuối của mạng NN.

    ![alt text](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%3D%5Cmathbf%7BW%7D%5E%7B%28l%29%7D%5Ccdot%5Cmathbf%7Ba%7D%5E%7B%28l-1%29%7D%20%5Ccr%20%26%5Cmathbf%7Ba%7D%5E%7B%28l%29%7D%3Df%28%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%29%20%5Cend%7Baligned%7D)
​	  
    + Tính đạo hàm theo z ở tầng ra.

    ![alt text](https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28L%29%7D%7D%7D%20%3D%20%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Ba%7D%5E%7B%28L%29%7D%7D%7D%5Cdfrac%7B%5Cpartial%7B%5Cmathbf%7Ba%7D%5E%7B%28L%29%7D%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28L%29%7D%7D%7D)

    + Tính các đạo hàm ngược lại

    ![alt text](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%7D%7D%20%26%3D%20%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l&plus;1%29%7D%7D%7D%5Cdfrac%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l&plus;1%29%7D%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Ba%7D%5E%7B%28l%29%7D%7D%7D%5Cdfrac%7B%5Cpartial%7B%5Cmathbf%7Ba%7D%5E%7B%28l%29%7D%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%7D%7D%20%5Ccr%20%26%20%3D%20%5Cbigg%28%5Cbig%28%5Cmathbf%7BW%7D%5E%7B%28l&plus;1%29%7D%5Cbig%29%5E%7B%5Cintercal%7D%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l&plus;1%29%7D%7D%7D%5Cbigg%29%5Cdfrac%7B%5Cpartial%7B%5Cmathbf%7Ba%7D%5E%7B%28l%29%7D%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%7D%7D%20%5Cend%7Baligned%7D)

    + Rút ra đạo hàm.

    ![alt text](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7BW%7D%5E%7B%28l%29%7D%7D%7D%20%26%3D%20%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%7D%7D%5Cdfrac%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%7D%7D%7B%5Cpartial%7B%5Cmathbf%7BW%7D%5E%7B%28l%29%7D%7D%7D%20%5Ccr%20%26%20%3D%20%5Cdfrac%7B%5Cpartial%7BJ%7D%7D%7B%5Cpartial%7B%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%7D%7D%5Cbig%28%5Cmathbf%7Ba%7D%5E%7B%28l-1%29%7D%5Cbig%29%5E%7B%5Cintercal%7D%20%5Cend%7Baligned%7D)

- Tối ưu W và b:
    + Tối ưu W: 

    *Đạo hàm của W được tính bằng cách nhân delta đầu ra với đầu vào*

    + Tối ưu b:

    *Đạo hàm của b tính bằng tổng của delta theo chiều dọc*