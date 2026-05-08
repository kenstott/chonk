# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Answer generation primitives: AnswerContext, PromptBuilder, AnswerGenerator, Answer."""

from ._context import AnswerContext
from ._prompt_builder import PromptBuilder
from ._answer import Answer, AnswerGenerator

__all__ = ["AnswerContext", "PromptBuilder", "Answer", "AnswerGenerator"]
