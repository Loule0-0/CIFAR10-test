import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# CIFAR-10类别名称
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 1. 数据加载与预处理
def load_and_preprocess_data():       
    """加载并预处理CIFAR-10数据"""
    # 加载数据
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    print(f"训练集大小: {x_train.shape}")
    print(f"测试集大小: {x_test.shape}")
    print(f"像素值范围: {x_train.min()}-{x_train.max()}")
    
    return (x_train, y_train), (x_test, y_test)

# 2. 构建CNN模型
def create_problematic_cnn():
    """创建故意有缺陷的CNN模型"""
    model = keras.Sequential([

        layers.Conv2D(16, (7, 7), strides=3, activation='relu', input_shape=(32, 32, 3)),
        

        layers.MaxPooling2D((4, 4)),
        

        layers.Flatten(),
        
        layers.Dense(2000, activation='relu'),  
        layers.Dense(50, activation='relu'),    
        layers.Dense(10, activation='softmax')  
    ])
    
    return model

# 3. 编译模型
def compile_problematic_model(model):
    """编译模型，使用不太合适的参数"""

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),  
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy']
    )
    return model

# 4. 训练模型
def train_problematic_model(model, x_train, y_train, x_test, y_test):
    """训练模型"""

    history = model.fit(
        x_train, y_train,
        batch_size=512,  
        epochs=5,        
        validation_data=(x_test, y_test),
        verbose=1
    )
    return history

# 5. 数据可视化
def visualize_data(x_train, y_train, num_samples=10):
    """可视化部分训练数据"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(x_train[i])
        axes[i].set_title(f'标签: {class_names[y_train[i][0]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 6. 评估模型
def evaluate_model(model, x_test, y_test):
    """评估模型性能"""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集损失: {test_loss:.4f}")
    return test_loss, test_accuracy

# 7. 绘制训练历史
def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 准确率曲线
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)
    
    # 损失曲线
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 8. 预测示例
def predict_samples(model, x_test, y_test, num_samples=8):
    """预测几个样本"""
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # 预测
        prediction = model.predict(x_test[idx:idx+1], verbose=0)
        predicted_class = np.argmax(prediction)
        true_class = y_test[idx][0]
        confidence = np.max(prediction)
        
        # 显示图像
        axes[i].imshow(x_test[idx])
        color = 'green' if predicted_class == true_class else 'red'
        axes[i].set_title(f'真实: {class_names[true_class]}\n预测: {class_names[predicted_class]}\n置信度: {confidence:.2f}', 
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    print("开始训练有缺陷的CIFAR-10分类模型...")
    print("注意：这个模型故意包含多个问题，需要你来发现并改进")
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 可视化数据
    print("\n数据可视化:")
    visualize_data(x_train, y_train)
    
    # 创建并编译模型
    model = create_problematic_cnn()
    model = compile_problematic_model(model)
    
    # 显示模型结构
    print("\n模型结构:")
    model.summary()
    
    # 训练模型
    print("\n开始训练...")
    history = train_problematic_model(model, x_train, y_train, x_test, y_test)
    
    # 评估模型
    print("\n模型评估:")
    evaluate_model(model, x_test, y_test)
    
    # 可视化结果
    print("\n训练过程可视化:")
    plot_training_history(history)
    
    print("\n预测示例:")
    predict_samples(model, x_test, y_test)
    
    return model, history

if __name__ == "__main__":
    model, history = main()
