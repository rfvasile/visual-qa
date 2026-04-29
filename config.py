from enum import StrEnum


class Repositories(StrEnum):
    VQAGenerationEasyVQA = "rfvasile/blip2-easyvqa-gen"
    VQAGenerationDaquar = "rfvasile/blip2-daquar-gen"
    VQAClassificationEasyVQA = "rfvasile/blip2-easyvqa-classifier"
    VQAClassificationDaquar = "rfvasile/blip2-daquar-classifier"
