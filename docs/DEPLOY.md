# Publishing the repository and the project page

The project page is `docs/index.html` with its assets in `docs/assets/` (every link relative; `docs/.nojekyll` makes
GitHub Pages serve the folder as is). It opens locally from the file, without a server.

Repository: `https://github.com/noahcao/FAH`; page: `https://noahcao.github.io/FAH/`. The two URLs appear only in the
page's "Code" button (`docs/index.html`) and in `README.md` (the `git clone` command and the link definitions at
the end). To use another owner or repository name (and use those names in the steps below):

```bash
sed -i.bak -e "s|noahcao/FAH|OWNER/REPO|g" -e "s|noahcao.github.io/FAH|OWNER.github.io/REPO|g" README.md docs/index.html
rm README.md.bak docs/index.html.bak
```

The repository starts private. GitHub Pages is turned on only when it goes public: a Pages site is public even when
its repository is private (and Pages for private repositories needs a paid plan).

Steps:

1. Create an empty private repository `noahcao/FAH` on GitHub (no README, licence or .gitignore), on the website or
   with the GitHub CLI:
   ```bash
   gh repo create noahcao/FAH --private --description "FAH: Fast and Accurate Hand Estimation"
   ```
2. Push:
   ```bash
   cd FAH
   git remote add origin git@github.com:noahcao/FAH.git
   git push -u origin main
   ```
3. When the release goes public: make the repository public (Settings -> General -> Danger Zone -> Change visibility,
   or `gh repo edit noahcao/FAH --visibility public --accept-visibility-change-consequences`), then turn on GitHub
   Pages: Settings -> Pages -> Build and deployment -> Source "Deploy from a branch", Branch `main`, folder `/docs`
   -> Save. Or with the GitHub CLI:
   ```bash
   gh api -X POST repos/noahcao/FAH/pages -f "source[branch]=main" -f "source[path]=/docs"
   ```
4. After the first deployment (about a minute; Actions tab: "pages build and deployment"), open
   `https://noahcao.github.io/FAH/`. The account's Pages site uses a custom domain, so GitHub redirects this address to
   `https://www.jinkuncao.com/FAH/`; the README's link keeps working through the redirect.
