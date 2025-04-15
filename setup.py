from setuptools import setup
import os

PKG_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements() -> list:
    """Load requirements from file, parse them as a Python list."""

    with open(os.path.join(PKG_ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")
    install_requires = [str(x).strip() for x in all_reqs]

    return install_requires


setup(
    name='LangSpace',
    version='0.3.7',
    packages=['langspace', 'langspace.ops', 'langspace.probe', 'langspace.probe.sts', 'langspace.probe.defmod',
              'langspace.probe.lingprop', 'langspace.probe.traversal', 'langspace.probe.arithmetic',
              'langspace.probe.cluster_vis', 'langspace.probe.interpolation', 'langspace.probe.disentanglement',
              'langspace.models', 'langspace.metrics'],
    url='',
    license='',
    author=['Danilo S. Carvalho', 'Yingji Zhang'],
    author_email=['danilo.carvalho@manchester.ac.uk', 'yingji.zhang@postgrad.manchester.ac.uk'],
    description='LangSpace: Probing Large Language VAEs made simple',
    install_requires=load_requirements(),
)
