import pkgutil
import langchain
import langchain.chains as chains

print('langchain version', langchain.__version__)
print('chains path', chains.__path__)
print([m.name for m in pkgutil.iter_modules(chains.__path__)])
