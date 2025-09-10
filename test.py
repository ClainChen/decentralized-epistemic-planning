import pickle
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def diagnose_model_serialization(model_instance):
    """诊断Model实例的序列化问题"""
    print("=== Model序列化问题诊断 ===")
    
    problematic_attributes = []
    
    # 检查所有属性
    for attr_name in dir(model_instance):
        if attr_name.startswith('__'):
            continue
            
        try:
            attr_value = getattr(model_instance, attr_name)
            # 尝试序列化该属性
            pickle.dumps(attr_value)
        except Exception as e:
            problematic_attributes.append((attr_name, type(attr_value), str(e)))
    
    for attr_name, attr_type, error_message in problematic_attributes:
        print(f"属性 {attr_name} ({attr_type}) 存在序列化问题: {error_message}")
    return problematic_attributes