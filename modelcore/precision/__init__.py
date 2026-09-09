"""
modelcore.precision -- optional precision-changing transforms over a built Model. FP8 today
(fp8.py); the module boundary exists so a future precision scheme (e.g. int8 weight-only) has
somewhere to live without ModelManager growing a case per scheme.
"""
