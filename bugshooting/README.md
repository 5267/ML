### Bugshooting repo
When coders install some software, use some packages, bugs pop up often, this parts dedicates to record all the bugs i meet in learning and working process, good luck for no bug work and life.

1. error: command 'cl.exe' failed: No such file or directory
**Answer**: it's depending on some software you don't have installed. That looks like a Visual Studio component, so you may need to install that first. No compiler = Not going to happen. so what you need to do is to install Visual Studio compiler.(Python2.7就需要使用VS2008 C++ compiler ， 而python3至python3.4 采用VS2010 编译生成，而python3.5 须采用VS2015)

2. ImportError: cannot import name 'RLock'
在安装好catalyst库后(安装方法：直接git clone源码，使用setup.py方式安装，注意：python3.6之后的版本需要安装vs 2017），命令运行catalyst，出现以上报错，解决方案：pip uninstall gevent. 原因在于：catalyst库支持的新版本logbook版本与 gevent 存在不兼容情况，卸载 gevent 即可。

