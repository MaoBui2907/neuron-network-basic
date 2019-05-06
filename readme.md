Tạo neuron network cơ bản

++ Dữ liệu
- Dữ liệu X đầu vào là cặp 2 giá trị (tọa độ x và tọa độ y)
- Dữ liệu đầu ra là nhãn cho cặp dữ liệu là 0 hoặc 1 (red or blue)

++ Model
- dữ liệu đầu vào có 2 giá trị -> input layer có 2 nodes
- dữ liệu đầu ra có 2 giá trị là 0 hoặc 1 -> có thể chọn ouput layer là 1 node có giá trị 0,1 nhưng nên chọn 2 node để dễ dàng mở rộng sau này.

![alt text](http://www.wildml.com/wp-content/uploads/2015/09/nn-from-scratch-3-layer-network-1024x693.png)

- Chọn số chiều (số node) của hidden layer. Càng nhiều chiều, phương trình càng phức tạp và càng phù hợp với dữ liệu.  Nhưng càng nhiều chiều, cần tính toán càng nhiều hơn để đưa ra được dự đoán, cần nhiều network parameter, dễ bị tràn dữ liệu.

- Chọn activation function (là hàm phi tuyến xử lý input để cho ra output thuộc khoảng giá trị (0,1)) cho hidden layer. các loại activation function thường dùng là (tanh, sigmoid, ReLUs). Ở đây dùng sigmoid function.

- Sử dụng thêm hàm softmax để xử lý giá trị ra theo kiểu xác suất

