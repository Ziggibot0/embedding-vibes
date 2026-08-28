import time, asyncio
from gen_sessions import _gen_one
async def main():
    t0=time.time()
    try:
        steps = await asyncio.wait_for(_gen_one("ad hominem", "universal basic income"), timeout=90)
        print("OK %.1fs -> %d steps"%(time.time()-t0, len(steps)))
        for s in steps[:2]: print("  ", s[:70])
    except Exception as e:
        print("FAIL %.1fs: %r"%(time.time()-t0, str(e)[:160]))
asyncio.run(main())
