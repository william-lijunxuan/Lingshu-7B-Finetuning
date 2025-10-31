from typing import Any

class VLMRegistry:
    _models = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name):
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry")
        return cls._models[name]

@VLMRegistry.register("MedVLM-R1")
class MedVLM_R1:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from MedVLM_R1.MedVLM_R1 import MedVLM_R1
        return MedVLM_R1(model_path, args)

@VLMRegistry.register("LingShu")
class LingShu:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from qwenvl.eval.VLM.Lingshu.LingShu import LingShu
        return LingShu(model_path, args)

@VLMRegistry.register("MedGemma")
class MedGemma:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from Medgemma.MedGemma import MedGemma
        return MedGemma(model_path, args)


def init_llm(args):
    try:
        model_class = VLMRegistry.get_model(args.model_name)
        return model_class(args.model_path, args)
    except ValueError as e:
        raise ValueError(f"{args.model_name} not supported") from e

