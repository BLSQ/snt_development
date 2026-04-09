# Best Practices for SNT Configuration Management

Hi! Here are some ideas to implement a more robust approach to the way we work with SNT config files.
Here below I put some suggestions, looking forward to your feedback!

Please refer to this **Jira task**: https://bluesquare.atlassian.net/browse/SNT25-453

## 1. Use a JSON Schema

A JSON Schema is a powerful tool to define the structure and data types of your JSON files. It allows to validate config files against a predefined schema, ensuring that they are correctly formatted and contain the expected fields.

How to implement it:

- Keep "**snt_config.schema.json**" in the same folder as your config files.
- Ensure the "**SNT_config_XXX.json**" files have the `$schema` property:
  `"$schema": "./snt_config.schema.json"`.
  This will force VS Code to show red squiggles if a user puts a string where a number is expected.

_I took the liberty to already add these bits, hope they don't break anything ... !_

**FYI**: The URL http://json-schema.org/draft-07/schema# is the **Meta-Schema**. It doesn't actually "download" anything while the code runs; rather, it tells software (like VS Code or a Python validator) which "version" of the JSON Schema language you are using.<br>
This is equivalent to a `<!DOCTYPE html>` tag: it ensures the rules for validation are interpreted correctly.

## 2. Placeholders in the config template "SNT_config_XXX.json"

The generic version of the config file ("**SNT_config_XXX.json**") now contains all the mandatry fields, and none of the country-specific fields. The following placeholders are used to help filling up these fields with the correct data type:

- `null` for missing numbers/floats.
- `""` for missing mandatory strings.
- `[]` for missing lists of IDs.

## 3. (not mandatory but it helps) Use same JSON code formatter

In VS Code, I'm using [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) to ensure JSON files are automatically formatted at save. This way, spacing and line breaks etc are consistent and we don't get things like blank spaces being flagged as changes in git diff.

---

## (idea) Architectural Split: a More Robust and Intuitive Approcah

Instead of having one large file that users edit, we should consider splitting them into 2 files:

- `SNT_config_global.json`: Contains only the "Fixed" components (like SNT\*DATASET_IDENTIFIERS). The **user** should **not** be **exposed** to this because this is structural (do not touch!).
- `SNT_config_COD.json`: Contains only the country-specific fields. In this case I'm using DRC (COD) as a concrete example. The **user** must be able to correctly **modify** this file.

I put an example of each in the folder `./new_approach_idea`. It will require updating all the existing pipelines BUT I reckon it will improve everyone's experince around the config file ... less things breaking, more clarity for the user.
