import torch
import numpy as np
from mongoGNN import select_environment, AGENT_TYPES

print("🤖 Multi-Agent Safety Controller Training")
print("=" * 50)

# Test environment selection
agent = select_environment()
print(f"Selected agent: {agent}")

# Test basic imports work
from mongoGNN import SafetyGNN, SafetyController
print("✓ Core classes imported successfully")

print("\n✅ Basic functionality test passed!")
