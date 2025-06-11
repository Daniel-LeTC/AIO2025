# %% [markdown]
"""
<a href="https://colab.research.google.com/github/Daniel-LeTC/AIO2025/blob/main/M1W1/%5BColab-Hint%5D-Exercise-Activation-Functions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %% [markdown]
"""
## **Bài 1 - Viết hàm tính độ đo F1**
Viết hàm thực hiện đánh giá F1-Score cho các mô hình phân loại.
- $\mbox{Precision} = \dfrac{TP}{TP + FP}$
- $\mbox{Recall} = \dfrac{TP}{TP + FN}$
- $\mbox{F1-score} = 2*\dfrac{Precision*Recall}{Precision + Recall}$

- Input: nhận 3 giá trị **tp, fp, fn**

- Output: trả về kết quả của **Precision, Recall, và F1-score**


**NOTE: Đề bài yêu cầu các điều kiện sau**
    
- Phải **kiểm tra giá trị nhận vào tp, fp, fn là kiểu dữ liệu int**, nếu là nhận được kiểm dữ liệu khác khác thì in thông báo cho người dùng ví dụ check fn là float, print **'fn must be int'** và thoát hàm hoặc dừng chương trình.
- Yêu cầu **tp, fp, fn phải đều lớn hơn 0**, nếu không thì print **'tp and fp and fn must be greater than 0'** và thoát hàm hoặc dừng chương trình
"""

# %% [markdown]
"""
### Câu hỏi 1
"""

# %%
import math
def calc_f1_score(tp, fp, fn):
    """
    Tính điểm F1 Score từ các giá trị:
    - tp: Số lượng true positives
    - fp: Số lượng false positives
    - fn: Số lượng false negatives

    Công thức:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    Trả về:
        f1_score: Điểm F1 Score dưới dạng float
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision )
    return f1_score

assert round(calc_f1_score(tp=2, fp=3, fn=5), 2) == 0.33

# %%
calc_f1_score(tp=2, fp=4, fn=5)
# %%
def evaluate_f1_components(tp, fp, fn):
    """
    Tính điểm F1 Score từ các giá trị:
    - tp: Số lượng true positives
    - fp: Số lượng false positives
    - fn: Số lượng false negatives

    Công thức:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    Trả về:
        f1_score: Điểm F1 Score dưới dạng float
    """

    # Kiểm tra kiểu dữ liệu
    # Kiểm tra giá trị không âm
    ### Your code here
    assert all(isinstance(val, int) and val >= 0 for val in [tp, fp, fn])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

precision, recall, f1_score = evaluate_f1_components(tp=2, fp=3, fn=5)

# %%
precision, recall, f1_score

# %% [markdown]
"""
# **Bài 2: Hàm kích hoạt**
"""

# %% [markdown]
"""
- Input:
    - Người dùng nhập giá trị **x**
    - Người dùng nhập tên **activation function chỉ có 3 loại (sigmoid, relu, elu)**
- Output: Kết quả **f(x)** (x khi đi qua actiavtion fucntion tương ứng theo activation function name). Ví dụ **nhập x=3, activation_function = 'relu'. Output: print 'relu: f(3)=3'**

**NOTE: Lưu ý các điều kiện sau:**
- Dùng function **is_number** được cung cấp sẵn để **kiểm tra x có hợp lệ hay không** (vd: x='10', is_number(x) sẽ trả về True ngược lại là False). Nếu **không hợp lệ print 'x must be a number' và dừng chương trình.**
- Kiểm tra **activation function name có hợp lệ hay không nằm trong 3 tên (sigmoid, relu, elu)**. **Nếu không print 'ten_function_user is not supported'** (vd người dùng nhập 'belu' thì print 'belu is not supportted')
- Convert **x** sang **float** type
- Thực hiện với theo công thức với activation name tương ứng. Print ra kết quả
- Dùng math.e để lấy số e
- $\alpha$ = 0.01
"""

# %% [markdown]
"""
## **Câu hỏi 2**
"""

# %%
import math
def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True
assert is_number(3) == 1.0
assert is_number('-2a') == 0.0
print(is_number(1))
print(is_number('n'))

# %% [markdown]
"""
## **Câu hỏi 3**
"""

# %%
def calc_relu(x):
    """
    Tính hàm ReLU:
    ReLU(x) = max(0, x)
    """
    if x<=0:
        result = 0.0
    else:
        result = x
    return float(result)

calc_relu(5)

# %% [markdown]
"""
## **Câu hỏi 4**
"""

# %%
import math

def calc_sig(x):
    """
    Tính hàm sigmoid: σ(x) = 1 / (1 + e^(-x))
    """
    ### Your code here
    return 1 / (1 + math.exp(-x))

assert round(calc_sig(3), 2)==0.95

calc_sig(1)

# %%
print(round(calc_sig(2), 2))

# %% [markdown]
"""
## **Câu hỏi 5**
"""

# %%
import math

def calc_elu(x):
    """
    Tính hàm ELU (Exponential Linear Unit):
    ELU(x) = x                 nếu x >= 0
           = alpha * (e^x - 1) nếu x < 0
    """
    ### Your code here
    if x <= 0:
        return 0.01*(math.exp(x) - 1)
    else:
        return x

assert round(calc_elu(1))==1

calc_elu(-4)

# %%
print(round(calc_elu(-1), 2))

# %% [markdown]
"""
## **Câu hỏi 6**
"""

# %% [markdown]
"""
Do em không đọc kĩ file notebook nên tưởng là tự làm nguyên phần câu 2 vào đây nên em code lại từ đầu :D có hỏi AI một số cái và code lại nên kết quả của câu 6 chạy hàm sigmoid bằng 0.95 là ở 2 cell kế tiếp ạ, mong TA thông cảm :D
"""

# %%
def relu(x:float) -> float:
    if x < 0:
        return 0
    else:
        return x

def elu(x:float, alpha:float) -> float:
    if x <= 0:
        return alpha*(math.exp(x) - 1)
    else:
        return x
def sigmoid(x:float) -> float:
    return 1 / (1 + math.exp(-x))

def activation_function(name_func: str, x:float) -> float:
    activation_funcs = {
        'relu': relu,
        'sigmoid': sigmoid,
        'elu': elu
    }
    if not is_number(x):
        raise ValueError(f'{x} must be a number')
    if name_func not in activation_funcs:
        raise ValueError(f'Error:{name_func} not supported')
    call_func = activation_funcs[name_func]
    if name_func == 'elu':
        alpha = 0.01
        return call_func(x, alpha)
    else:
        return call_func(x)

# %% [markdown]
"""
###### sigmoid_output
"""

# %%
# Test 1:
round(activation_function('sigmoid', 3),2)

# %%
# Test 2
activation_function('elu1')

# %%
# Test 3
activation_function('elu', 'n')

# %%
import math

def calc_activation_func(x, act_name):
    """
    Tính hàm kích hoạt cho x dựa trên act_name:
    'relu', 'sigmoid', hoặc 'elu'.
    """
    ### Your code here

assert calc_activation_func(x = 1, act_name='relu') == 1
calc_activation_func(1, "sigmoid")

# %%
print(round(calc_activation_func(x = 3, act_name='sigmoid'), 2))


# %% [markdown]
"""
##### Please see the code and output of this sigmoid output at [output](#sigmoid_output)

"""

# %%
def interactive_activation_function():
    x = input('Input x = ')
    if not is_number(x):
        print('x must be a number')
        return # exit()

    act_name = input('Input activation function (sigmoid|relu|elu): ')
    x = float(x)
    result = calc_activation_func(x, act_name)
    if result is None:
        print(f'{act_name} is not supportted')
    else:
        print(f'{act_name}: f({x}) = {result}')

# %%
interactive_activation_function()

# %% [markdown]
"""
# **Bài 3: Hàm mất mát**
"""

# %% [markdown]
"""
Viết function lựa chọn regression loss function để tính loss:
- MAE = $ \dfrac{1}{n}∑_{i=1}^{n} |y_{i} - \hat{y}_{i}| $
- MSE = $ \dfrac{1}{n}∑_{i=1}^{n} (y_{i} - \hat{y}_{i})^2 $
- RMSE = $\sqrt{MSE}$ = $ \sqrt{\dfrac{1}{n}∑_{i=1}^{n} (y_{i} - \hat{y}_{i})^2} $
- **n** chính là **số lượng samples (num_samples)**, với **i** là mỗi sample cụ thể. Ở đây các bạn có thể hiểu là cứ mỗi **i** thì sẽ **có 1 cặp  $y_i$ là target và $\hat{y}$ là predict**.
- Input:
    - Người dùng **nhập số lượng sample (num_samples) được tạo ra (chỉ nhận integer numbers)**
    - Người dùng **nhập loss name (MAE, MSE, RMSE-(optional)) chỉ cần MAE và MSE, bạn nào muốn làm thêm RMSE đều được**.
        
- Output:
    - Print ra **loss name, sample, predict, target, loss**
        - **loss name:** là loss mà người dùng chọn
        - **sample:** là thứ tự sample được tạo ra (ví dụ num_samples=5, thì sẽ có 5 samples và mỗi sample là sample-0, sample-1, sample-2, sample-3, sample-4)
        - **predict:** là số mà model dự đoán (chỉ cần dùng random tạo random một số trong range [0,10))
        - **target:** là số target mà momg muốn mode dự đoán đúng (chỉ cần dùng random tạo random một số trong range [0,10))
        - **loss:** là kết quả khi đưa predict và target vào hàm loss
        - **note:** ví dụ num_sample=5 thì sẽ có 5 cặp predict và target.

**Note: Các bạn lưu ý**
- Dùng **.isnumeric() method** để kiểm tra **num_samples** có hợp lệ hay không (vd: x='10', num_samples.isnumeric() sẽ trả về True ngược lại là False). Nếu **không hợp lệ print 'number of samples must be an integer number'** và dừng chương trình.
- **Dùng vòng lặp for, lặp lại num_samples lần**. **Mỗi lần dùng random modules tạo một con số ngẫu nhiên trong range [0.0, 10.0) cho predict và target**. Sau đó predict và target vào loss function và print ra kết quả mỗi lần lặp.
- Dùng **random.uniform(0,10)** để tạo ra một số ngẫu nhiên trong range [0,10)
- **Giả xử người dùng luôn nhập đúng loss name MSE, MAE, và RMSE (đơn giảng bước này để các bạn không cần check tên hợp lệ)**
- Dùng abs() để tính trị tuyệt đối ví dụ abs(-3) sẽ trả về 3
- Dùng math.sqrt() để tính căn bậc 2
"""

# %% [markdown]
"""
## **Câu hỏi 7**
"""

# %%
ef calc_ae(y, y_hat):
    """
    Tính sai số tuyệt đối (Absolute Error)
    giữa giá trị thực tế và giá trị dự đoán.

    Tham số:
    y (float hoặc int): Giá trị thực tế (ground truth).
    y_hat (float hoặc int): Giá trị dự đoán.

    Trả về:
    float: Giá trị tuyệt đối của hiệu giữa y và y_hat.
    """
    ### Your code here
    return abs(y - y_hat)

y = 1
y_hat = 6
assert calc_ae(y, y_hat)==5

y = 2
y_hat = 9
print(calc_ae(y, y_hat))

# %% [markdown]
"""
## **Câu hỏi 8**
"""

# %%
def calc_se(y, y_hat):
    """
    Tính sai số bình phương (Squared Error)
    giữa giá trị thực tế và giá trị dự đoán.

    Tham số:
    y (float hoặc int): Giá trị thực tế (ground truth).
    y_hat (float hoặc int): Giá trị dự đoán.

    Trả về:
    float: Bình phương của hiệu giữa y và y_hat.
    """
    ### Your code here
    return (y - y_hat)**2
y = 4
y_hat = 2
assert calc_se(y, y_hat) == 4

print(calc_se(2, 1))

# %%
import random

def cal_activation_function():
    num_samples = input('Input number of samples (integer number) which are generated: ')
    if not num_samples.isnumeric():#Hàm isnumeric() trong Python trả về true nếu một chuỗi dạng Unicode chỉ chứa các ký tự số,
    #nếu không là false.
        print("number of samples must be an integer number")
        return # exit()
    loss_name = input('Input loss name: ')

    # giả sử người dùng luôn nhập đúng MAE, MSE hoặc RMSE
    final_loss = 0
    num_samples = int(num_samples)
    for i in range(num_samples):
        pred_sample = random.uniform(0,10)
        target_sample = random.uniform(0,10)

        if loss_name == 'MAE':
            loss = calc_ae(pred_sample, target_sample)
        elif loss_name == 'MSE' or loss_name == 'RMSE':
            loss = calc_se(pred_sample, target_sample)
        # hoặc trả về thông báo loss không có
        final_loss += loss
        print(f'loss_name: {loss_name}, sample: {i}: pred: {round(pred_sample,2)} target: {round(target_sample,2)} loss: {round(loss,2)}')

    final_loss /= num_samples
    if loss_name == 'RMSE':
        final_loss = math.sqrt(final_loss)
    print(f'final {loss_name}: {final_loss}')
    return final_loss

# %%
final_loss = cal_activation_function()
final_loss

# %% [markdown]
"""
# **Bài 4: Hàm lượng giác**
"""

# %% [markdown]
"""
Viết 4 functions để ước lượng các hàm số sau.
-  Input: x (số muốn tính toán) và n (số lần lặp muốn xấp xỉ)
- Output: Kết quả ước lượng hàm tương ứng với x. Ví dụ hàm cos(x=0) thì output = 1

**NOTE: Các bạn chú ý các điều kiện sau**
- x là radian
- n là số nguyên dương > 0
- các bạn nên viết một hàm tính giai thừa riêng
"""

# %%
def factorial_fcn(x):
    """
    Compute the factorial of a non-negative integer x.

    Parameters:
    x (int): The input integer (x >= 0)

    Returns:
    int: The factorial of x (i.e., x!)
    """
    res = 1
    for i in range(x):
        res *= (i + 1)
    return res

factorial_fcn(x=4)

# %% [markdown]
"""
## **Câu hỏi 9**
"""

# %%
def approx_sin(x, n):
    """
    Approximate the sine of x using the Taylor series expansion.

    Parameters:
    x (float): The input angle in radians.
    n (int): Number of terms in the Taylor series expansion.

    Returns:
    float: Approximate value of sin(x) using n+1 terms.
    """
    ### Your code here
    sin_approx = 0.0
    for i in range(n+1):
        expo = 2 * i + 1
        denom = factorial_fcn(expo)

        sign = (-1)**i

        term = sign * (x**expo/denom)
        sin_approx += term

    return sin_approx

# %%
print(round(approx_sin(x=3.14, n=10), 4))

# %% [markdown]
"""
## **Câu hỏi 10**
"""

# %%
def approx_cos(x, n):
    """
    Approximate the cosine of x using the Taylor series expansion.
    Parameters:
    x (float): The input angle in radians.
    n (int): Number of terms in the Taylor series expansion.
    Returns:
    float: Approximate value of cos(x) using n+1 terms.
    """
    ### Your code here
    cos_approx = 0.0
    for i in range(n+1):
        expo_c = 2 * i
        denom_c = factorial_fcn(expo_c)
        sign_c = (-1)**i
        term = sign_c * (x**expo_c/denom_c)
        cos_approx += term
    return cos_approx



# Test
assert round(approx_cos(x=1, n=10), 2) == 0.54

# %%
print(round(approx_cos(x=3.14, n=10), 2))

# %% [markdown]
"""
## **Câu hỏi 11**
"""

# %%
def approx_sinh(x, n):
    """
    Approximate the hyperbolic sine of x using the Taylor series expansion.
    Parameters:
    x (float): The input value.
    n (int): Number of terms in the Taylor series expansion.
    Returns:
    float: Approximate value of sinh(x) using n+1 terms.
    """
    ### Your code here
    sinh_approx = 0.0
    for i in range(n+1):
        expo = 2 * i + 1
        denom = factorial_fcn(expo)

        sign = (-1)**i

        term = (x**expo)/denom
        sinh_approx += term
    return sinh_approx
# Test
assert round(approx_sinh(x=1, n=10), 2) == 1.18

# %%
print(round(approx_sinh(x=3.14, n=10), 2))

# %% [markdown]
"""
## **Câu hỏi 12**
"""

# %%
def approx_cosh(x, n):
    """
    Approximate the hyperbolic cosine of x using the Taylor series.
    Parameters:
    x (float): The input value.
    n (int): Number of terms in the Taylor series expansion.
    Returns:
    float: Approximate value of cosh(x) using n+1 terms.
    """
    ### Your code here
    cosh_approx = 0.0
    for i in range(n+1):
        expo_c = 2 * i
        denom_c = factorial_fcn(expo_c)
        # sign_c = (-1)**i
        term = (x**expo_c)/denom_c
        cosh_approx += term
    return cosh_approx

# Test
assert round(approx_cosh(x=1, n=10), 2) == 1.54

# %%
print(round(approx_cosh(x=3.14, n=10), 2))

