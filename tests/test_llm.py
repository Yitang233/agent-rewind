"""Token accounting: a SQLite cache hit is charged what the original call cost."""
import unittest


class CachedTokensTest(unittest.TestCase):
    def test_disk_cache_hit_is_charged_the_original_tokens(self):
        from agent_rewind.llm import CachedLLM

        class Hit:                        # what LlamaCppLLM reports after a sqlite cache hit
            last_tokens = 0
            last_stats = {"cached_call": True, "orig_prefill": 300, "orig_decoded": 40, "prefill": 0, "decoded": 0}

            def complete(self, role, task, inputs):
                return {"x": 1}

        _, used = CachedLLM(Hit()).complete("r", "t", {"a": 1})
        self.assertEqual(used, 340)


if __name__ == "__main__":
    unittest.main()
