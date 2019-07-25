class PHOTONPrintHelper:
    @staticmethod
    def _optimize_printing(pipe, config: dict):
        """
        make the sklearn config syntax prettily readable for humans
        """
        if pipe is None:
            return str(config)

        prettified_config = ["" + '\n']
        for el_key, el_value in config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in pipe.named_steps:
                new_pretty_key = '    ' + name + '->'
                prettified_config.append(new_pretty_key +
                                         pipe.named_steps[name].prettify_config_output(rest, el_value) + '\n')
            else:
                raise ValueError('Item is not contained in pipeline:' + name)
        return ''.join(prettified_config)

    @staticmethod
    def config_to_human_readable_dict(pipe, specific_config):
        """
        """
        prettified_config = {}
        for el_key, el_value in specific_config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in pipe.named_steps:
                prettified_config[name] = pipe.named_steps[name].prettify_config_output(rest, el_value)
            else:
                raise ValueError('Item is not contained in pipeline:' + name)
        return prettified_config
