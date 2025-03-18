# 股票分析系统 (Stock Analysis System)
关于量化分析，不建议使用，仅供学习参考，请勿用于实盘交易。

## 简介
基于 https://github.com/lanzhihong6/stock-scanner 二次开发，感谢原作者

## 功能
- [x] 支持登陆/注册
- [x] 服务端配置多 llm 模型
- [x] 通过线程池优化 llm 调用
- [x] 动态更新 akshare
- [x] 技术分析
- [ ] DCF模型

## docker compose 部署
- 修改 .env.example 文件为 .env，替换对应的值，然后执行以下命令启动容器：
- 修改 docker-compose.example.yml 文件为 docker-compose.yml，然后执行以下命令启动容器：

```bash
docker compose up -d
```


## 许可证 (License)
[待添加具体许可证信息]

## 免责声明 (Disclaimer)
本系统仅用于学习和研究目的，投资有风险，入市需谨慎。
