# Use Node.js 20 LTS (Iron) as the base image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

ENV NODE_ENV=production

CMD [ "executable" ]
