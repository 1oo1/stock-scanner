# Use Node.js 20 LTS (Iron) as the base image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Set environment to production for optimized build
ENV NODE_ENV=production

# No CMD needed as this is just a build environment
# Container can be used with docker run commands to execute build steps
