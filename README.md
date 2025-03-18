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
- 注意：数据库是通过依赖的形式添加的，所以 .env 的数据库 host 是 db 容器的容器名。如果有其他postgresql 数据库，请修改对应的 host 和 port。

```bash
docker compose up -d
```

## nginx 反向代理
```
server {
	listen 443 ssl;
	listen [::]:443 ssl;

	server_name your.domain;

	ssl_certificate /etc/ssl/certs/your.domain.pem; # Path to your certificate
  ssl_certificate_key /etc/ssl/private/your.domain.key; # Path to your private key

	location / {
		proxy_pass http://127.0.0.1:8888/;
		proxy_buffering off;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-Prefix /;
	}
}
```

## 许可证 (License)
[待添加具体许可证信息]

## 免责声明 (Disclaimer)
本系统仅用于学习和研究目的，投资有风险，入市需谨慎。
