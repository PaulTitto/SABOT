import asyncio, os
from lightrag.llm.gemini import gemini_embed as _wrapped

async def test():
    from lightrag.utils import TokenTracker
    t = TokenTracker()
    result = await _wrapped.func(
        ['test text hello world'],
        api_key=os.getenv('GEMINI_API_KEY'),
        model='gemini-embedding-001',
        embedding_dim=1536,
        token_tracker=t,
    )
    print('Usage:', t.get_usage())

if __name__ == "__main__":
    asyncio.run(test())
