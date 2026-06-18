"""
Synthetic Dialogue Memory Search Benchmark.

This benchmark generates long conversation logs containing:
  1. An oracle sentence with a specific item (e.g., "The user mentioned they purchased a crimson sedan yesterday.")
  2. Unrelated distractor sentences (e.g., "The user was planning to buy a blue jacket.")
  3. A query that uses a synonym/hypernym of the target item (e.g., "What color vehicle did the user buy?")
  4. A reference answer (e.g., "crimson").

This exposes the exact vocabulary mismatch problem that limits grep, while letting us
control the number of distractor sentences to stress-test context rot in vector search.
"""
import random

CATEGORIES = {
    "vehicle": {
        "specifics": ["sedan", "coupe", "motorcycle", "suv", "truck", "hatchback"],
        "template": "The user mentioned they purchased a {color} {item} yesterday.",
        "distractor_template": "The user was looking at a {color} {item} online.",
        "query": "What color vehicle did the user buy?",
        "distractors": ["jacket", "bicycle", "umbrella", "backpack", "sofa"]
    },
    "occupation": {
        "specifics": ["physician", "software engineer", "barista", "plumber", "cardiologist", "mechanic"],
        "template": "The user got a new job working as a {color} {item}.",
        "distractor_template": "The user's cousin works as a {color} {item}.",
        "query": "What occupation did the user get?",
        "distractors": ["artist", "teacher", "chef", "writer", "photographer"]
    },
    "residence": {
        "specifics": ["condo", "loft", "cottage", "penthouse", "bungalow", "cabin"],
        "template": "The user moved into a beautiful {color} {item} downtown.",
        "distractor_template": "The user rented a {color} {item} for their vacation.",
        "query": "What type of residence did the user move into?",
        "distractors": ["apartment", "house", "villa", "hotel room", "office"]
    },
    "pet": {
        "specifics": ["cocker spaniel", "siamese cat", "parakeet", "hamster", "guinea pig", "goldfish"],
        "template": "The user adopted a cute {color} {item} from the shelter.",
        "distractor_template": "The user saw a {color} {item} at the pet store.",
        "query": "What type of pet did the user adopt?",
        "distractors": ["dog", "cat", "rabbit", "turtle", "lizard"]
    },
    "refreshment": {
        "specifics": ["espresso", "iced tea", "lemonade", "kombucha", "smoothie", "macchiato"],
        "template": "The user ordered a cold {color} {item} at the cafe.",
        "distractor_template": "The user spilled their {color} {item} on the table.",
        "query": "What type of refreshment did the user order?",
        "distractors": ["water", "coffee", "tea", "soda", "juice"]
    },
    "apparel": {
        "specifics": ["cardigan", "trousers", "blazer", "sneakers", "trenchcoat", "pullover"],
        "template": "The user bought a matching {color} {item} for the party.",
        "distractor_template": "The user washed their {color} {item} in hot water.",
        "query": "What piece of apparel did the user buy?",
        "distractors": ["shirt", "pants", "shoes", "hat", "socks"]
    }
}

COLORS = ["crimson", "emerald", "sapphire", "golden", "indigo", "violet", "amber", "charcoal", "scarlet", "turquoise"]
DISTRACTOR_COLORS = ["red", "blue", "green", "yellow", "black", "white", "gray", "orange", "brown", "pink"]

CHATTER = [
    "The weather was quite nice today, perfect for walking.",
    "The user mentioned they are planning a trip to Chicago next month.",
    "They spent the afternoon reading a science fiction novel.",
    "The assistant suggested trying out a new Italian restaurant nearby.",
    "They discussed upgrading their laptop to a faster processor.",
    "The user complained about a mild headache from staring at screens.",
    "The assistant provided a recipe for homemade sourdough bread.",
    "They talked about the upcoming football match on Sunday evening.",
    "The user wants to start learning piano online next week.",
    "The conversation touched on the benefits of drinking green tea.",
]


def generate_example(category_name: str, num_distractors: int, seed: int = 42) -> dict:
    """Generate a single dialogue context history, query, and reference answer."""
    rng = random.Random(seed)
    cat = CATEGORIES[category_name]
    
    # 1. Oracle target
    target_color = rng.choice(COLORS)
    target_item = rng.choice(cat["specifics"])
    oracle_sentence = cat["template"].format(color=target_color, item=target_item)
    
    # 2. Build turns
    turns = []
    
    # Add random chatter
    for _ in range(num_distractors // 2):
        turns.append(rng.choice(CHATTER))
        
    # Add target oracle
    turns.append(oracle_sentence)
    
    # Add target distractors (same category but different item/color)
    for _ in range(min(3, num_distractors // 4)):
        dist_color = rng.choice(DISTRACTOR_COLORS)
        dist_item = rng.choice([x for x in cat["specifics"] if x != target_item])
        turns.append(cat["distractor_template"].format(color=dist_color, item=dist_item))
        
    # Add other categories' templates as distractors
    for _ in range(num_distractors - len(turns)):
        other_cat_name = rng.choice([c for c in CATEGORIES.keys() if c != category_name])
        other_cat = CATEGORIES[other_cat_name]
        dist_color = rng.choice(DISTRACTOR_COLORS)
        dist_item = rng.choice(other_cat["specifics"])
        turns.append(other_cat["template"].format(color=dist_color, item=dist_item))
        
    # Shuffle dialogue turns to distribute oracle randomly
    rng.shuffle(turns)
    
    # Format as a chat transcript
    dialogue_history = []
    for idx, turn in enumerate(turns):
        role = "User" if idx % 2 == 0 else "Assistant"
        dialogue_history.append(f"[{role}]: {turn}")
        
    context = "\n".join(dialogue_history)
    
    return {
        "category": category_name,
        "context": context,
        "query": cat["query"],
        "reference_answer": target_color if category_name == "vehicle" else target_item,
        "target_item": target_item
    }


def generate_benchmark_dataset(num_examples_per_cat: int = 5, num_distractors: int = 15, base_seed: int = 100) -> list[dict]:
    """Generate a full dataset containing multiple examples across all categories."""
    dataset = []
    idx = 0
    for cat_name in CATEGORIES.keys():
        for i in range(num_examples_per_cat):
            ex = generate_example(cat_name, num_distractors, seed=base_seed + idx)
            ex["id"] = idx
            dataset.append(ex)
            idx += 1
    return dataset
