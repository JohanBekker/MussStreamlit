from transformers import BartForConditionalGeneration

def Convert2HalfPrecision(load_path, target_path):
    model = BartForConditionalGeneration.from_pretrained(load_path)
    model.half()
    model.save_pretrained(target_path)
    return