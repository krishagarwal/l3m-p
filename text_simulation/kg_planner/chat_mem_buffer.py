from __future__ import annotations
import random
from typing import Any, List
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer


class TripletTrimBuffer(ChatMemoryBuffer):
    def get(self, input: str, initial_token_count: int = 0, *, triplets: str, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        chat_history = self.get_all()

        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")
        
        token_count = self._token_count_for_messages(chat_history)
        if token_count >= self.token_limit:
            return super().get()
        
        triplet_list = triplets.split("\n")
        remove_idx = 0
        remaining_count = self.token_limit - token_count
        curr_message = ChatMessage.from_str(input.format(filtered_triplet_str=triplets))
        token_count = self._token_count_for_messages([curr_message])
        while self._token_count_for_messages([curr_message]) > remaining_count:
            triplet_list.pop(remove_idx)
            remove_idx = (remove_idx + 5) % len(triplet_list)
            curr_message = ChatMessage.from_str(input.format(filtered_triplet_str="\n".join(triplet_list)))
        return [curr_message] + chat_history