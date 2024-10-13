import yaml

# 读取yaml配置文件
with open('/root/Tip-Adapter/configs/breakhis.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 打印配置
print(config)

# 使用配置中的参数
root_path = config['root_path']
dataset_name = config['dataset']
shots = config['shots']