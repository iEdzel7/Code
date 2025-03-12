    async def request_configuration(hass, config_entry, login):
        """Request configuration steps from the user using the configurator."""

        async def configuration_callback(callback_data):
            """Handle the submitted configuration."""
            await hass.async_add_job(
                setup_platform_callback, hass, config_entry, login, callback_data
            )

        configurator = hass.components.configurator
        status = login.status
        email = login.email
        # links = ""
        footer = ""
        config_id = None
        if status and status.get("error_message"):
            footer = (
                "\n<b>NOTE: Actual Amazon error message in red below. "
                "Remember password will be provided automatically"
                " and Amazon error message normally appears first!</b>"
            )
        # if login.links:
        #     links = '\n\nGo to link with link# (e.g. link0)\n' + login.links
        # Get Captcha
        if status and status.get("captcha_image_url"):
            _LOGGER.debug("Creating configurator to request captcha")
            config_id = configurator.async_request_config(
                f"Alexa Media Player - Captcha - {email}",
                configuration_callback,
                description=(
                    "Please enter the text for the captcha."
                    " Please enter anything to reload image."
                    # + links
                    + footer
                ),
                description_image=status["captcha_image_url"],
                submit_caption="Confirm",
                fields=[{"id": "captcha", "name": "Captcha"}],
            )
        elif status and status.get("securitycode_required"):  # Get 2FA code
            _LOGGER.debug("Creating configurator to request 2FA")
            config_id = configurator.async_request_config(
                f"Alexa Media Player - 2FA - {email}",
                configuration_callback,
                description=(
                    "Please enter your Two-Factor Security code."
                    # + links
                    + footer
                ),
                submit_caption="Confirm",
                fields=[{"id": "securitycode", "name": "Security Code"}],
            )
        elif status and status.get("claimspicker_required"):  # Get picker method
            _LOGGER.debug("Creating configurator to select verification option")
            options = status["claimspicker_message"]
            if options:
                config_id = configurator.async_request_config(
                    f"Alexa Media Player - Verification Method - {email}",
                    configuration_callback,
                    description=(
                        "Please select the verification method by number. "
                        "(e.g., `0` or `1`).\n{}".format(options)
                        # + links
                        + footer
                    ),
                    submit_caption="Confirm",
                    fields=[{"id": "claimsoption", "name": "Option"}],
                )
            else:
                await configuration_callback({})
        elif status and status.get("authselect_required"):  # Get picker method
            _LOGGER.debug("Creating configurator to select OTA option")
            options = status["authselect_message"]
            if options:
                config_id = configurator.async_request_config(
                    f"Alexa Media Player - OTP Method - {email}",
                    configuration_callback,
                    description=(
                        "Please select the OTP method by number. "
                        "(e.g., `0`, `1`).<br />{}".format(options)
                        # + links
                        + footer
                    ),
                    submit_caption="Confirm",
                    fields=[{"id": "authselectoption", "name": "Option"}],
                )
            else:
                await configuration_callback({})
        elif status and status.get("verificationcode_required"):  # Get picker method
            _LOGGER.debug("Creating configurator to enter verification code")
            config_id = configurator.async_request_config(
                f"Alexa Media Player - Verification Code - {email}",
                configuration_callback,
                description=(
                    "Please enter received verification code."
                    # + links
                    + footer
                ),
                submit_caption="Confirm",
                fields=[{"id": "verificationcode", "name": "Verification Code"}],
            )
        else:  # Check login
            _LOGGER.debug("Creating configurator to start new login attempt")
            config_id = configurator.async_request_config(
                f"Alexa Media Player - Begin - {email}",
                configuration_callback,
                description=("Please hit confirm to begin login attempt."),
                submit_caption="Confirm",
                fields=[],
            )
        if config_id:
            hass.data[DATA_ALEXAMEDIA]["accounts"][email]["configurator"].append(
                config_id
            )
        if status.get("error_message"):
            configurator.async_notify_errors(config_id, status["error_message"])
        if status.get("message"):
            configurator.async_notify_errors(config_id, status["message"])
        if len(hass.data[DATA_ALEXAMEDIA]["accounts"][email]["configurator"]) > 1:
            configurator.async_request_done(
                (hass.data[DATA_ALEXAMEDIA]["accounts"][email]["configurator"]).pop(0)
            )