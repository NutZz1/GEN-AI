"""LLM reasoning/planning/verification layer on top of the AvatarBlend VAE.

The VAE (avatarblend.py, traits.py) stays the generative core; nothing here retrains or bypasses it.
See agent/agent.py for the Plan -> Act -> Verify -> Self-correct -> Explain loop.
"""
