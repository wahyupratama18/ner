import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        outDir: 'static',
        rollupOptions: {
            input: './src/app.js',
            output: {
                entryFileNames: 'js/app.js',
                assetFileNames: (assetInfo) => {
                    if (assetInfo.name.endsWith('.css')) {
                        return 'css/[name][extname]';
                    }
                    return 'assets/[name][extname]';
                }
            }
        }
    }
});